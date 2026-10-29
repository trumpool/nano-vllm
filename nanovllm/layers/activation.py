# =============================================================================
# activation.py —— SwiGLU 激活：SiLU(gate) × up
#
# MLP 里 gate_proj 和 up_proj 被合并成一个矩阵乘（gate_up_proj），输出 [N, 2 × intermediate]，
# 这里把它切成两半：前一半是 gate，后一半是 up。
# SiLU(x) = x × sigmoid(x)。@torch.compile 把 chunk、silu、乘法融合成一个 kernel。
# =============================================================================
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
