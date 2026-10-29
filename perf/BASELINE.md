# Baseline on RTX 5070 Ti (sm_120)

Day 1 of the project: get nano-vllm running on a consumer Blackwell GPU and measure where it stands.

## Environment (2026-10-28)

| | |
|---|---|
| GPU | RTX 5070 Ti, 16 GB GDDR7, sm_120, 70 SMs; theoretical 896 GB/s, 46.4 TFLOPS FP32 |
| Driver | 610.88 (Windows 11, WSL2, Ubuntu 24.04) |
| nano-vllm | upstream `bb823b3` (2026-04-26, after the chunked-prefill refactor) |
| torch / triton | 2.8.0+cu128 / 3.4.0 |
| flash-attn | 2.8.3.post2, prebuilt wheel `cu12torch2.8cxx11abiTRUE-cp312` |
| model | Qwen3-0.6B (bf16; 28 layers, 16 q heads, 8 kv heads, head_dim 128) |

### Setup

flash-attn 2.8.3.post2 only ships prebuilt wheels for torch <= 2.8 (CUDA 12), so this env pins torch 2.8:

```bash
python3 -m venv ~/venvs/nanovllm && source ~/venvs/nanovllm/bin/activate
pip install "torch==2.8.*" --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install "transformers>=4.51.0" xxhash numpy
pip install -e . --no-deps
```

The prebuilt flash-attn wheel works on sm_120: `flash_attn_varlen_func` (with and without `block_table`) and paged `flash_attn_with_kvcache` match a PyTorch reference within bf16 tolerance (max abs err 0.016 / 0.002 / 0.001).

## Offline throughput (`bench.py`)

Workload: `bench.py` as-is. 256 sequences, input and output lengths uniform in [100, 1024], `ignore_eos`, seed 0, 133,966 output tokens total. Each number is one full run after the warmup call in `bench.py`.

| Engine | Config | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|---|
| nano-vllm | defaults (`gpu_memory_utilization=0.9`, CUDA graphs) | 4547 | 4523 | 4559 | **4543 tok/s** |
| SGLang 0.5.22 | defaults | 3862 | 3940 | 3938 | 3913 tok/s |
| SGLang 0.5.22 | `mem_fraction_static=0.85`, `cuda_graph_max_bs_decode=256` | 4485 | 4562 | 4558 | **4535 tok/s** |

SGLang ran the identical prompts and lengths (same RNG call order) through `sgl.Engine.generate(input_ids=...)`; script: [`perf/bench_sglang.py`](bench_sglang.py).

For reference, the upstream README reports 1434 tok/s on an RTX 4070 Laptop for the same workload.

### Notes

- **nano-vllm matches a tuned SGLang on this workload.** On the defaults SGLang is ~14% slower, because on this GPU it only captures decode CUDA graphs up to batch size 8 and reserves less memory for KV cache. With 256 concurrent sequences most decode steps then run without CUDA graphs.
- This is a pure offline throughput test: all requests arrive at once. It says nothing about latency (TTFT, TPOT) under online load, where request arrival patterns and scheduling policy matter.
- `Scheduler.schedule()` never mixes prefill and decode in one step: if any sequence is waiting for prefill, the step is prefill-only and running sequences don't advance. Worth measuring under online load (day 3).
