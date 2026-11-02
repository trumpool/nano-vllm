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
    # "auto": KV cache in the model dtype. "fp8": float8_e4m3fn, half the bytes per token; decode attention runs on
    # FlashInfer, which reads fp8 directly (flash-attn 2 can't)
    kv_cache_dtype: str = "auto"
    # fp8 only: per-tensor (k, v) scales, cache = clamp(x / scale, +-448). Qwen3-0.6B's layer-0 keys reach ~490,
    # past e4m3's max of 448; values stay below ~130
    kv_cache_scales: tuple[float, float] = (2.0, 1.0)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.kv_cache_dtype in ("auto", "fp8")
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
