import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


def paged_kv_layout(block_tables: list[list[int]], seq_lens: list[int], block_size: int, batch_size: int = 0):
    # FlashInfer's paged KV format: the block tables concatenated (indices), each sequence's offset into them (indptr),
    # and how many tokens its last block holds (last_page_len). batch_size > len(seq_lens) pads with dummy requests
    # that read 1 token of block 0 (CUDA graph padding; their outputs are dropped)
    indptr, indices, last_page_len = [0], [], []
    for block_table, seq_len in zip(block_tables, seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        indices.extend(block_table[:num_blocks])
        indptr.append(len(indices))
        last_page_len.append(seq_len - (num_blocks - 1) * block_size)
    for _ in range(batch_size - len(seq_lens)):
        indices.append(0)
        indptr.append(len(indices))
        last_page_len.append(1)
    return indptr, indices, last_page_len


def plan_attention(wrapper, layout, num_qo_heads: int, num_kv_heads: int, head_dim: int, block_size: int,
                   q_dtype: torch.dtype, kv_dtype: torch.dtype, qo_indptr: list[int] | None = None):
    # decode wrappers: one query per sequence. Prefill wrappers (qo_indptr): causal, the queries are the last tokens
    # of each sequence's KV
    indptr, indices, last_page_len = layout
    args = (torch.tensor(indptr, dtype=torch.int32, device="cpu"),
            torch.tensor(indices, dtype=torch.int32, device="cpu", pin_memory=True).cuda(non_blocking=True),
            torch.tensor(last_page_len, dtype=torch.int32, device="cpu"),
            num_qo_heads, num_kv_heads, head_dim, block_size)
    kwargs = dict(q_data_type=q_dtype, kv_data_type=kv_dtype, sm_scale=head_dim ** -0.5)
    if qo_indptr is None:
        wrapper.plan(*args, **kwargs)
    else:
        wrapper.plan(torch.tensor(qo_indptr, dtype=torch.int32, device="cpu"), *args, causal=True, **kwargs)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.use_fp8 = config.kv_cache_dtype == "fp8"
        self.kv_dtype = torch.float8_e4m3fn if self.use_fp8 else hf_config.dtype
        if self.use_fp8:
            import flashinfer
            # allocated before warmup so that the KV cache sizing (peak memory) accounts for them
            self.flashinfer_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8)
            self.flashinfer_int_workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)
            self.decode_wrapper = self.flashinfer_wrapper(flashinfer.BatchDecodeWithPagedKVCacheWrapper, use_tensor_cores=True)
            self.prefill_wrapper = self.flashinfer_wrapper(flashinfer.BatchPrefillWithPagedKVCacheWrapper)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.kv_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, dtype=self.kv_dtype)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if self.use_fp8:
                    module.k_scale, module.v_scale = config.kv_cache_scales
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def flashinfer_wrapper(self, cls, **kwargs):
        # all wrappers share both workspaces (FlashInfer would give each its own 8 MB int workspace, x37 CUDA graphs):
        # only one runs at a time, and it is re-planned right before it runs
        wrapper = cls(self.flashinfer_workspace, "NHD", **kwargs)
        wrapper.reset_workspace_buffer(self.flashinfer_workspace, self.flashinfer_int_workspace)
        return wrapper

    def plan(self, wrapper, layout, qo_indptr: list[int] | None = None):
        hf_config = self.config.hf_config
        plan_attention(wrapper, layout, hf_config.num_attention_heads // self.world_size,
                       hf_config.num_key_value_heads // self.world_size, hf_config.head_dim, self.block_size,
                       hf_config.dtype, self.kv_dtype, qo_indptr)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        attn_wrapper = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            if self.use_fp8:    # flash-attn can't read the fp8 cache, FlashInfer's paged prefill can
                seq_lens = [seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs]
                self.plan(self.prefill_wrapper, paged_kv_layout([seq.block_table for seq in seqs], seq_lens, self.block_size), cu_seqlens_q)
                attn_wrapper = self.prefill_wrapper
            else:
                block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, attn_wrapper)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if self.use_fp8:    # planned in run_model, once we know which wrapper (eager or which CUDA graph) runs
            block_tables = None
            self.decode_seqs = ([seq.block_table for seq in seqs], [len(seq) for seq in seqs])
        else:
            block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if not is_prefill and self.use_fp8:
                self.plan(self.decode_wrapper, paged_kv_layout(*self.decode_seqs, self.block_size))
                get_context().attn_wrapper = self.decode_wrapper
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph_bs = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[graph_bs]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            if self.use_fp8:
                # fills the wrapper's buffers (and the shared workspaces) that the graph reads
                self.plan(self.graph_wrappers[graph_bs], paged_kv_layout(*self.decode_seqs, self.block_size, graph_bs))
            else:
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
        self.graph_wrappers = {}

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            wrapper = None
            if self.use_fp8:
                # one wrapper per graph: the captured kernels read its fixed-address buffers, which plan() refills
                import flashinfer
                wrapper = self.flashinfer_wrapper(
                    flashinfer.BatchDecodeWithPagedKVCacheWrapper, use_cuda_graph=True, use_tensor_cores=True,
                    paged_kv_indptr_buffer=torch.zeros(bs + 1, dtype=torch.int32),
                    paged_kv_indices_buffer=torch.zeros(bs * max_num_blocks, dtype=torch.int32),
                    paged_kv_last_page_len_buffer=torch.zeros(bs, dtype=torch.int32),
                )
                self.plan(wrapper, paged_kv_layout([], [], self.block_size, bs))    # capture with all-padding requests
                self.graph_wrappers[bs] = wrapper
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], attn_wrapper=wrapper)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
