# =============================================================================
# config.py —— 引擎的所有配置
#
#   max_num_batched_tokens  一步最多处理多少个 token（限制 prefill 的规模，也决定 warmup 测显存时用多大的 batch）
#   max_num_seqs            一步最多多少个序列（也决定 CUDA graph 录到多大的 batch，上限 512）
#   max_model_len           单个序列的最大长度（不超过模型本身支持的长度）
#   gpu_memory_utilization  最多用总显存的多少（剩下的给系统和其他程序）
#   enforce_eager           True = 不用 CUDA graph（调试时方便，但 decode 会慢）
#   kvcache_block_size      每个 KV block 存多少个 token。必须是 256 的倍数：
#                           这是 flash-attn 2 的 paged KV cache 对 block 大小的要求
#   num_kvcache_blocks      KV block 总数，-1 表示由 model_runner 量完显存后自动算出
# =============================================================================
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)     # model 必须是本地目录（不会自动从 HF 下载）
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
