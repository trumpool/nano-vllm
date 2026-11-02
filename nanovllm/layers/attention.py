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
    k_scale,
    v_scale,
    D: tl.constexpr,
    FP8: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    if FP8:
        # saturate: e4m3fn has no inf, and a PyTorch cast turns out-of-range values into NaN. Triton's cast happens to
        # saturate already (satfinite), the clamp makes that explicit rather than an accident of the code generator
        key = tl.clamp(key.to(tl.float32) / k_scale, -448.0, 448.0)
        value = tl.clamp(value.to(tl.float32) / v_scale, -448.0, 448.0)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key.to(k_cache_ptr.dtype.element_ty))
    tl.store(v_cache_ptr + cache_offsets, value.to(v_cache_ptr.dtype.element_ty))


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor,
                  k_scale: float = 1.0, v_scale: float = 1.0):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    fp8 = k_cache.dtype == torch.float8_e4m3fn
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, k_scale, v_scale, D, fp8)


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
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = 1.0    # fp8 KV cache: real value = cached value * scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.k_scale, self.v_scale)
        if context.attn_wrapper is not None:    # fp8 decode / prefix-cache prefill: FlashInfer reads the fp8 cache directly
            o = context.attn_wrapper.run(q, (k_cache, v_cache), k_scale=self.k_scale, v_scale=self.v_scale)
        elif context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
