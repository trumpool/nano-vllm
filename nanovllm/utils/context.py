# =============================================================================
# context.py —— 用一个全局变量，把"这一步的 attention 元数据"传给模型深处的每一层
#
# 为什么要这样：attention 层需要 cu_seqlens、slot_mapping、block_tables 这些信息，
# 但它们和模型结构无关。如果作为参数一层层传（model → layer → attention），每个 forward 的签名都要改。
# 所以 model_runner 在每一步开始时 set_context(...)，attention 层里 get_context() 直接取，
# 这一步跑完 reset_context()。vLLM 里也有类似的 "forward context" 机制。
#
# 字段：
#   is_prefill     这一步是 prefill 还是 decode
#   cu_seqlens_q/k prefill 用：各序列 query / key 的起止位置（前缀和），varlen attention 需要
#   max_seqlen_q/k prefill 用：最长序列的长度
#   slot_mapping   每个新 token 的 K/V 写到 KV cache 的哪个位置
#   context_lens   decode 用：每个序列当前总长度
#   block_tables   每个序列的 KV 在哪些 block 里（decode 一定有；prefill 只在有 prefix cache 时有）
# =============================================================================
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
