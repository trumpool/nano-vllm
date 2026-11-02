"""store_kvcache: the Triton kernel that writes each step's K/V into the paged cache (bf16, or fp8 with scale + clamp)."""
import pytest
import torch

from nanovllm.layers.attention import store_kvcache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_BLOCKS, BLOCK, KV_HEADS, DIM = 8, 256, 8, 128


def store(k, v, slots, cache_dtype, k_scale=1.0, v_scale=1.0):
    k_cache = torch.zeros(NUM_BLOCKS, BLOCK, KV_HEADS, DIM, dtype=cache_dtype, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    store_kvcache(k, v, k_cache, v_cache, torch.tensor(slots, dtype=torch.int32, device="cuda"), k_scale, v_scale)
    return k_cache.view(-1, KV_HEADS, DIM), v_cache.view(-1, KV_HEADS, DIM)


def random_kv(n, std=1.0):
    torch.manual_seed(0)
    k = torch.randn(n, KV_HEADS, DIM, dtype=torch.bfloat16, device="cuda") * std
    v = torch.randn(n, KV_HEADS, DIM, dtype=torch.bfloat16, device="cuda") * std
    return k, v


def to_fp8(x, scale):    # reference quantization
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def test_bf16_store_is_exact():
    k, v = random_kv(100)
    slots = torch.randperm(NUM_BLOCKS * BLOCK)[:100].tolist()
    k_cache, v_cache = store(k, v, slots, torch.bfloat16)
    assert torch.equal(k_cache[slots], k) and torch.equal(v_cache[slots], v)


@pytest.mark.parametrize("k_scale, v_scale", [(1.0, 1.0), (2.0, 1.0), (0.37, 1.9)])
def test_fp8_store_matches_reference_quantization(k_scale, v_scale):
    k, v = random_kv(300, std=50)
    slots = torch.randperm(NUM_BLOCKS * BLOCK)[:300].tolist()
    k_cache, v_cache = store(k, v, slots, torch.float8_e4m3fn, k_scale, v_scale)
    # compare bit patterns: fp8 has no NaN-free equality issues this way, and 0 vs -0 would be a real difference
    assert torch.equal(k_cache[slots].view(torch.uint8), to_fp8(k, k_scale).view(torch.uint8))
    assert torch.equal(v_cache[slots].view(torch.uint8), to_fp8(v, v_scale).view(torch.uint8))


def test_fp8_store_saturates_instead_of_nan():
    # e4m3fn has no inf: a plain cast of anything past 448 gives NaN. The kernel must clamp
    k, v = random_kv(16, std=1000)
    k_cache, v_cache = store(k, v, list(range(16)), torch.float8_e4m3fn)
    assert not k_cache.float().isnan().any() and not v_cache.float().isnan().any()
    assert k_cache.float().abs().max() == 448


def test_padding_slots_are_skipped():
    k, v = random_kv(4)
    k_cache, _ = store(k, v, [5, -1, 7, -1], torch.bfloat16)
    written = k_cache.flatten(1).abs().sum(1).nonzero().flatten().tolist()
    assert written == [5, 7]
