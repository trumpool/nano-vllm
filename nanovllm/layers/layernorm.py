# =============================================================================
# layernorm.py —— RMSNorm，以及"残差相加 + RMSNorm"的融合版本
#
# RMSNorm(x) = x / sqrt(mean(x²) + eps) × weight     （和 LayerNorm 比：不减均值、没有 bias）
#
# Transformer 每层里有两次"先加残差、再归一化"。add_rms_forward 把这两件事合在一个函数里，
# 加上 @torch.compile，编译后变成一个 kernel：读一次 x 和 residual、写一次结果，省掉中间结果的显存读写。
# 计算在 float32 里做（平方和容易在 bf16 里损失精度），最后再转回 bf16。
# =============================================================================
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回两个东西：
        #   x        = RMSNorm(x + residual)   → 送进下一个子层（attention 或 MLP）
        #   residual = x + residual            → 新的残差，继续往后传
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
