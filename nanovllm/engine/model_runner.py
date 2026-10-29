# =============================================================================
# model_runner.py —— 在 GPU 上真正跑模型的地方
#
# 初始化（只做一次）：
#   1. 建模型、加载权重
#   2. warmup_model：用最大的 prefill batch 跑一次，量出"激活值最多占多少显存"
#   3. allocate_kv_cache：剩下的显存全部拿来做 KV cache，切成 block
#   4. capture_cudagraph：给 decode 的各种 batch size 各录一张 CUDA graph
#
# 每一步 run(seqs, is_prefill)：
#   input_ids, positions = prepare_prefill(seqs) 或 prepare_decode(seqs)
#       ↑ 把 Python 的 Sequence 对象拍平成 GPU 张量，并把 attention 需要的元数据放进全局 Context
#   logits = run_model(input_ids, positions)
#       ↑ prefill：直接跑（eager）；decode：把输入拷进固定缓冲区，replay CUDA graph
#   token_ids = sampler(logits, temperatures).tolist()
#       ↑ .tolist() 把结果拷回 CPU —— 这里 CPU 会等 GPU 跑完，是每一步的同步点
#
# 两个核心概念：
#   slot：KV cache 里"一个 token 的位置"，slot = block_id * block_size + 块内偏移
#         slot_mapping[i] 告诉 attention 层：本步第 i 个 token 的 K/V 要写到哪个 slot
#   varlen：prefill 时把多个序列的 token 首尾相接拼成一维，不做 padding；
#           用 cu_seqlens（前缀和）记录每个序列的起止位置，例如长度 [3, 5] → cu_seqlens = [0, 3, 8]
# =============================================================================
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


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 就算只有一张卡也会初始化 NCCL 进程组（world_size=1），这样 TP 相关代码不用到处写 if
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        # 临时把默认 dtype 设成模型的 dtype（bf16）、默认设备设成 cuda，
        # 这样下面建模型、建 KV cache 时创建的张量直接就是 GPU 上的 bf16
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            # TP：rank 0 建共享内存，其他 rank 连上去后进入 loop() 等指令（永远不会从构造函数返回）
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

    # ---------------- Tensor Parallel 的进程间通信 ----------------
    # rank 0 调用 call("run", seqs, ...) 时：先把 (方法名, 参数) pickle 进共享内存并通知其他 rank，
    # 再在本地执行；其他 rank 在 loop() 里读出来执行同一个方法。这样所有 rank 每一步都跑同样的前向。
    # 共享内存的格式：[4 字节长度 n][n 字节 pickle 数据]

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

    # ---------------- 初始化：测显存、分配 KV cache ----------------

    def warmup_model(self):
        # 用"一步最多能跑的 prefill"跑一次（默认 16384 个 token = 4 个 4096 长度的序列），
        # 让 PyTorch 分配出最大的激活值显存，后面用 peak 统计来算还剩多少显存给 KV cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)     # 这些假序列没有 block_table，attention 层不会读写 KV cache
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free                                             # 整张卡已用（含 CUDA context、权重、其他进程）
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]    # PyTorch 曾经最多分配过多少（权重 + 最大激活）
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # PyTorch 现在分配着多少（权重）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 一个 block 的字节数 = 2 (K 和 V) × 层数 × 256 个 token × KV 头数 × head_dim × 每个数的字节数
        # Qwen3-0.6B：2 × 28 × 256 × 8 × 128 × 2 B = 28 MiB（每个 token 112 KiB）
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 可用 = 总显存 × 0.9 − 已用 − (峰值 − 当前)
        #   "已用"里已经包含了权重（current），再减去激活值的峰值增量（peak − current），给前向留够空间
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 一整块大张量：[K/V, 层, block, block 内位置, KV 头, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 把每一层对应的切片（view，不复制）挂到那一层的 Attention 模块上
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # ---------------- 每一步：准备输入 ----------------

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 把各个序列长短不一的 block_table 用 -1 补齐成一个矩阵 [num_seqs, max_blocks]
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 对每个序列：只算还没缓存的那一段 token[start:end]（start = 已缓存的 token 数）
        #   q 的长度 = end - start（本步要算的 token）
        #   k 的长度 = end（attention 要看到从头开始的所有 token，包括 prefix cache 命中的部分）
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))     # RoPE 用的位置编号
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # 算出 token[start:end] 各自的 slot：按 block 分段，每段是连续的一串 slot
            #   第一段从 start 在块内的偏移开始；最后一段到 end 在块内的位置结束
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
            # k 比 q 长 ⇒ 有序列命中了 prefix cache 或者是 chunked prefill 的后续段：
            # 前面那部分 K/V 已经在 cache 里了，attention 要通过 block_tables 去 cache 里读
            block_tables = self.prepare_block_tables(seqs)
        # pin_memory + non_blocking：先放进锁页内存，再异步拷到 GPU，CPU 不用等拷贝完成
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # decode：每个序列只算 1 个 token（上一步采样出来的 last_token）
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))          # attention 要看的 token 数（包括它自己）
            # 这个 token 的 K/V 写到最后一个 block 里的最后一个已用位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ---------------- 每一步：执行 ----------------

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # prefill、强制 eager、或 batch 超过 512：直接正常跑
        #   prefill 不用 CUDA graph：每步 token 数都不一样（没法给每种形状录 graph），
        #   而且 prefill 计算量大，kernel launch 的开销占比很小
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # decode：挑一个 batch size ≥ 当前 bs 的 graph（比如 bs=100 → 用 bs=112 的 graph）
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # CUDA graph 录制时用的是固定地址的缓冲区，所以每一步把新输入拷进这些缓冲区，再 replay
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # 多出来的 padding 位置：slot = -1 → store_kvcache 会跳过，不会写坏 KV cache
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # padding 位置 context_len = 0 → attention 不看任何 token，它们的输出是无用的，最后也会被切掉
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            # 只取前 bs 行；lm_head（compute_logits）在 graph 外面跑
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # 只有 rank 0 采样（TP 时 logits 已经 gather 到 rank 0）；.tolist() 是 GPU → CPU 的同步点
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        # CUDA graph：把一次 decode 前向里的几百个 kernel launch "录"下来，之后一次 replay 全部发出去，
        # 省掉每个 kernel 的 CPU launch 开销。decode 时每个 kernel 都很小，launch 开销占比很大，所以收益明显
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 固定地址的输入/输出缓冲区，所有 graph 共用（按最大 bs 分配）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 录这些 batch size：1, 2, 4, 8, 16, 32, 48, ..., 512
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小录：第一张（最大的）graph 分配显存池，后面小的 graph 复用同一个池，省显存
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
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
