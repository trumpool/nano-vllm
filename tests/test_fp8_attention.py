"""FP8 KV cache attention, as the engine runs it (Attention layer + FlashInfer wrappers planned by model_runner's
helpers), against a float32 reference on the dequantized cache.

- decode, eager
- decode under CUDA graphs: two graphs of different batch sizes share one workspace, are captured on all-padding
  requests and replayed after re-planning with real ones (the engine's invariant)
- prefix-cache prefill: queries are the last tokens of a longer sequence whose first part is already cached
"""
import pytest
import torch
import torch.nn.functional as F

from nanovllm.engine.model_runner import paged_kv_layout, plan_attention
from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.utils.context import reset_context, set_context

flashinfer = pytest.importorskip("flashinfer")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_BLOCKS, BLOCK, HEADS, KV_HEADS, DIM = 32, 256, 16, 8, 128    # Qwen3-0.6B's attention shapes
K_SCALE, V_SCALE = 2.0, 0.5


@pytest.fixture(scope="module")
def workspaces():
    return (torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"))


def make_wrapper(cls, workspaces, **kwargs):    # as ModelRunner.flashinfer_wrapper
    wrapper = cls(workspaces[0], "NHD", **kwargs)
    wrapper.reset_workspace_buffer(*workspaces)
    return wrapper


def plan(wrapper, layout, qo_indptr=None):
    plan_attention(wrapper, layout, HEADS, KV_HEADS, DIM, BLOCK, torch.bfloat16, torch.float8_e4m3fn, qo_indptr)


class Batch:
    """Sequences with random block tables; the first `cached` tokens of each are already in the cache, the next
    `new` tokens are this step's queries (their K/V are written by the attention layer itself)."""

    def __init__(self, attn, cached, new, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.seq_lens = [c + n for c, n in zip(cached, new)]
        perm = torch.randperm(NUM_BLOCKS, generator=torch.Generator().manual_seed(seed)).tolist()
        self.block_tables = [[perm.pop() for _ in range((n + BLOCK - 1) // BLOCK)] for n in self.seq_lens]
        slots = [[t[i // BLOCK] * BLOCK + i % BLOCK for i in range(n)] for t, n in zip(self.block_tables, self.seq_lens)]
        k = [torch.randn(n, KV_HEADS, DIM, device="cuda", generator=g).bfloat16() * 4 for n in self.seq_lens]
        v = [torch.randn(n, KV_HEADS, DIM, device="cuda", generator=g).bfloat16() for n in self.seq_lens]
        self.q = torch.cat([torch.randn(n, HEADS, DIM, device="cuda", generator=g).bfloat16() for n in new])
        attn.k_cache.zero_(), attn.v_cache.zero_()
        old = lambda xs: torch.cat([x[:c] for x, c in zip(xs, cached)])
        if sum(cached):
            store_kvcache(old(k), old(v), attn.k_cache, attn.v_cache,
                          torch.tensor(sum((s[:c] for s, c in zip(slots, cached)), []), dtype=torch.int32, device="cuda"),
                          K_SCALE, V_SCALE)
        self.k = torch.cat([x[c:] for x, c in zip(k, cached)])
        self.v = torch.cat([x[c:] for x, c in zip(v, cached)])
        self.slot_mapping = torch.tensor(sum((s[c:] for s, c in zip(slots, cached)), []), dtype=torch.int32, device="cuda")
        self.new = new
        self.qo_indptr = [0]
        for n in new:
            self.qo_indptr.append(self.qo_indptr[-1] + n)

    def layout(self, batch_size=0):
        return paged_kv_layout(self.block_tables, self.seq_lens, BLOCK, batch_size)

    def reference(self, attn):
        # float32 attention over the dequantized cache, causal with the queries at the end of each sequence
        out = []
        k_cache = attn.k_cache.float() * K_SCALE
        v_cache = attn.v_cache.float() * V_SCALE
        for i, (table, n, nq) in enumerate(zip(self.block_tables, self.seq_lens, self.new)):
            k = k_cache[table].flatten(0, 1)[:n].repeat_interleave(HEADS // KV_HEADS, 1).transpose(0, 1)
            v = v_cache[table].flatten(0, 1)[:n].repeat_interleave(HEADS // KV_HEADS, 1).transpose(0, 1)
            q = self.q[self.qo_indptr[i]:self.qo_indptr[i + 1]].float().transpose(0, 1)
            mask = torch.arange(n, device="cuda")[None, :] <= torch.arange(n - nq, n, device="cuda")[:, None]
            out.append(F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=DIM ** -0.5).transpose(0, 1))
        return torch.cat(out)


@pytest.fixture
def attn():
    attn = Attention(HEADS, DIM, DIM ** -0.5, KV_HEADS)
    attn.k_cache = torch.zeros(NUM_BLOCKS, BLOCK, KV_HEADS, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    attn.v_cache = torch.zeros_like(attn.k_cache)
    attn.k_scale, attn.v_scale = K_SCALE, V_SCALE
    yield attn
    reset_context()


def check(out, ref):
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


DECODE_LENS = [1, 255, 256, 257, 700, 1500]    # around block boundaries; context length incl. the new token


@torch.inference_mode()
def test_decode_eager(attn, workspaces):
    batch = Batch(attn, [n - 1 for n in DECODE_LENS], [1] * len(DECODE_LENS), seed=0)
    wrapper = make_wrapper(flashinfer.BatchDecodeWithPagedKVCacheWrapper, workspaces, use_tensor_cores=True)
    plan(wrapper, batch.layout())
    set_context(False, slot_mapping=batch.slot_mapping, attn_wrapper=wrapper)
    out = attn(batch.q, batch.k, batch.v)
    check(out, batch.reference(attn))


@torch.inference_mode()
def test_decode_cuda_graphs_replanned_with_shared_workspace(attn, workspaces):
    max_blocks = 8
    graphs = {}
    for bs in (8, 4):
        wrapper = make_wrapper(
            flashinfer.BatchDecodeWithPagedKVCacheWrapper, workspaces, use_cuda_graph=True, use_tensor_cores=True,
            paged_kv_indptr_buffer=torch.zeros(bs + 1, dtype=torch.int32, device="cuda"),
            paged_kv_indices_buffer=torch.zeros(bs * max_blocks, dtype=torch.int32, device="cuda"),
            paged_kv_last_page_len_buffer=torch.zeros(bs, dtype=torch.int32, device="cuda"))
        plan(wrapper, paged_kv_layout([], [], BLOCK, bs))    # capture on all-padding requests, like the engine
        static = dict(q=torch.zeros(bs, HEADS, DIM, dtype=torch.bfloat16, device="cuda"),
                      k=torch.zeros(bs, KV_HEADS, DIM, dtype=torch.bfloat16, device="cuda"),
                      v=torch.zeros(bs, KV_HEADS, DIM, dtype=torch.bfloat16, device="cuda"),
                      slot_mapping=torch.full((bs,), -1, dtype=torch.int32, device="cuda"))
        set_context(False, slot_mapping=static["slot_mapping"], attn_wrapper=wrapper)
        attn(static["q"], static["k"], static["v"])    # warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static["out"] = attn(static["q"], static["k"], static["v"])
        reset_context()
        graphs[bs] = graph, wrapper, static

    # alternate between the graphs: each replay must see its own plan, not the other graph's
    for seed, (lens, bs) in enumerate([(DECODE_LENS[:5], 8), ([3, 600, 1024], 4), (DECODE_LENS[1:], 8), ([512], 4)]):
        batch = Batch(attn, [n - 1 for n in lens], [1] * len(lens), seed=seed)
        graph, wrapper, static = graphs[bs]
        plan(wrapper, batch.layout(bs))
        n = len(lens)
        static["q"][:n], static["k"][:n], static["v"][:n] = batch.q, batch.k, batch.v
        static["slot_mapping"].fill_(-1)
        static["slot_mapping"][:n] = batch.slot_mapping
        graph.replay()
        check(static["out"][:n], batch.reference(attn))


@torch.inference_mode()
def test_prefix_cache_prefill(attn, workspaces):
    # (cached, new): no prefix at all, block-aligned prefix, unaligned prefix + chunk crossing a block, 1 new token
    cached, new = [0, 256, 300, 1000, 37], [100, 256, 250, 1, 600]
    batch = Batch(attn, cached, new, seed=1)
    wrapper = make_wrapper(flashinfer.BatchPrefillWithPagedKVCacheWrapper, workspaces)
    plan(wrapper, batch.layout(), batch.qo_indptr)
    set_context(True, slot_mapping=batch.slot_mapping, attn_wrapper=wrapper)
    out = attn(batch.q, batch.k, batch.v)
    check(out, batch.reference(attn))
