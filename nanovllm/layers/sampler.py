# =============================================================================
# sampler.py —— 按温度从 logits 采样出下一个 token
#
# 用的是 Gumbel-max 技巧的一个变体（"exponential race"）：
#   想要按概率 p 抽一个 token，可以给每个 token 抽一个 E_i ~ Exp(1)，然后取 argmax(p_i / E_i)。
#   这和按 p 抽样在数学上完全等价，但只需要逐元素运算 + 一个 argmax，不需要 torch.multinomial
#   （multinomial 要算前缀和、二分查找，在 GPU 上慢，而且会引入同步）。
#
#   logits / T → softmax → p → p / Exp(1) 噪声 → argmax
# =============================================================================
import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile      # 让 torch.compile 把这几个逐元素运算融合成少数几个 kernel
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits: [batch, vocab]；temperatures: [batch]，每个序列可以有自己的温度
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # clamp_min 防止除以 0
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
