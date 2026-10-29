# =============================================================================
# rotary_embedding.py —— RoPE（旋转位置编码）
#
# 思路：把 head_dim 维的向量看成 head_dim/2 个二维平面上的点，第 j 个平面按 "位置 × θ_j" 旋转一个角度。
#   θ_j = base^(-2j / head_dim)       j = 0, 1, ..., head_dim/2 - 1（base 对 Qwen3 是 1,000,000）
# 旋转之后，q·k 的结果只和两个 token 的"相对位置"有关，这就是 RoPE 编码相对位置的原理。
#
# 这里用的是 "NeoX 风格"：前一半维度 x1 和后一半维度 x2 配成一对（而不是相邻的两维配对）
#   y1 = x1·cos − x2·sin
#   y2 = x2·cos + x1·sin
#
# cos/sin 对所有位置预先算好存进 cos_sin_cache，前向时按 positions 直接查表。
# =============================================================================
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))   # θ_j，共 head_dim/2 个
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)       # [max_pos, head_dim/2]，第 (i, j) 个 = i × θ_j
        cos = freqs.cos()
        sin = freqs.sin()
        # [max_pos, 1, head_dim]：前一半是 cos，后一半是 sin；中间的 1 是为了在"头"这一维上广播
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]      # 每个 token 按它的位置查表
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    # lru_cache：参数一样就返回同一个对象 → 28 层共用一个 RotaryEmbedding，cos/sin 表只存一份
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
