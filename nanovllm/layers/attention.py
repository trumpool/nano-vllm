# =============================================================================
# attention.py —— 每一层的 attention：先把新 token 的 K/V 写进 KV cache，再算 attention
#
# 伪代码：
#   forward(q, k, v):                         # q/k/v 是本步所有新 token 的，已经过 RoPE
#       ctx = get_context()                   # model_runner 在 prepare_* 里放进去的元数据
#       if KV cache 已分配:
#           store_kvcache(k, v → cache[slot_mapping])
#       if prefill:
#           if 有 prefix cache / chunked prefill:  k, v = 整个 cache（通过 block_tables 读出前面已有的部分）
#           o = flash_attn_varlen_func(...)   # 多个长度不同的序列拼在一起，一次算完，causal mask
#       else:  # decode，每个序列 1 个 query
#           o = flash_attn_with_kvcache(...)  # paged attention：按 block_table 去 cache 里找这个序列的 K/V
#
# 注意：flash-attn 2 的 paged KV cache 要求 block 大小是 256 的倍数（所以 config.py 里有那条 assert）。
# =============================================================================
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    # 一个 program（相当于 CUDA 的一个 block）负责一个 token：
    # 把它的 K 和 V（各 D = num_kv_heads × head_dim 个数）拷到 cache 里的第 slot 个位置
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return       # -1 = CUDA graph 的 padding 位置，不写
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # cache 的形状是 [num_blocks, block_size, num_kv_heads, head_dim]，
    # 把前两维看成一维的"slot"，第 slot 个位置的起点就是 slot × D
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 这些 assert 保证内存布局是 kernel 假设的那样：每个 token 的 D 个数在内存里是连续的
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)   # grid = N 个 program


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 先放空张量占位；model_runner.allocate_kv_cache() 会把真正的 cache 切片挂上来
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q: [N, num_heads, head_dim]；k/v: [N, num_kv_heads, head_dim]；N = 本步所有新 token 的总数
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():      # warmup 时 cache 还没分配，跳过
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                # 前缀的 K/V 不在这一步的 k/v 里，只在 cache 里 → 直接把整个 cache 交给 flash-attn，
                # 它会按 block_table 找到每个序列的全部 K/V（新 token 的刚刚已经写进去了）
                k, v = k_cache, v_cache
            # varlen：q 按 cu_seqlens_q 切成各个序列，k/v 按 cu_seqlens_k 切；causal=True 是下三角 mask
            # GQA：16 个 q 头共用 8 个 kv 头（每 2 个 q 头共用 1 组 K/V），flash-attn 内部处理
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # q 变成 [batch, 1, num_heads, head_dim]：每个序列 1 个 query token
            # cache_seqlens 告诉它每个序列有多长，block_table 告诉它每个序列的 K/V 在哪些 block 里
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
