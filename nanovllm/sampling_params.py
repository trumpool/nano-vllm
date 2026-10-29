# =============================================================================
# sampling_params.py —— 每个请求的采样参数
#
#   temperature  温度：越小越确定，越大越随机。必须 > 0：
#                采样器要除以温度（layers/sampler.py），而且没有实现 greedy（argmax）分支
#   max_tokens   最多生成多少个 token
#   ignore_eos   True = 生成了 EOS 也不停，一直生成到 max_tokens（benchmark 用，保证输出长度固定）
# =============================================================================
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
