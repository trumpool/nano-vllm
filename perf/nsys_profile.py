"""A small bench.py-like run, annotated with NVTX ranges, for Nsight Systems.

Every engine step is an NVTX range ("prefill bs=.. tok=.." / "decode bs=.."), with nested ranges for schedule,
prepare, run_model, sample and postprocess, so the timeline shows what the CPU is doing while the GPU runs kernels.
Only the measured generate() is recorded (cudaProfilerStart/Stop): startup, warmup and CUDA graph capture are skipped.

    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --cuda-graph-trace=node -o perf/results/nsys_bf16 -f true python perf/nsys_profile.py
    nsys stats -r cuda_gpu_kern_sum,nvtx_sum perf/results/nsys_bf16.nsys-rep
"""
import argparse
import os
from random import randint, seed

import torch
import torch.cuda.nvtx as nvtx

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.layers.sampler import Sampler

parser = argparse.ArgumentParser()
parser.add_argument("--num-seqs", type=int, default=64)
parser.add_argument("--max-output", type=int, default=256)
parser.add_argument("--kv-cache-dtype", type=str, default="")
args = parser.parse_args()

seed(0)
extra = dict(kv_cache_dtype=args.kv_cache_dtype) if args.kv_cache_dtype else {}
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), enforce_eager=False, max_model_len=4096, **extra)
prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(args.num_seqs)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(16, args.max_output)) for _ in range(args.num_seqs)]


def ranged(name, fn):
    def wrapper(*a, **kw):
        nvtx.range_push(name)
        try:
            return fn(*a, **kw)
        finally:
            nvtx.range_pop()
    return wrapper


o_step, o_schedule = LLMEngine.step, Scheduler.schedule


def schedule(self):
    seqs, is_prefill = o_schedule(self)
    # name the enclosing step range now that we know what it is
    nvtx.mark(f"{'prefill' if is_prefill else 'decode'} bs={len(seqs)} tok={sum(s.num_scheduled_tokens for s in seqs)}")
    return seqs, is_prefill


LLMEngine.step = ranged("step", o_step)
Scheduler.schedule = ranged("schedule", schedule)
Scheduler.postprocess = ranged("postprocess", Scheduler.postprocess)
ModelRunner.prepare_prefill = ranged("prepare_prefill", ModelRunner.prepare_prefill)
ModelRunner.prepare_decode = ranged("prepare_decode", ModelRunner.prepare_decode)
ModelRunner.run_model = ranged("run_model", ModelRunner.run_model)
Sampler.forward = ranged("sample", Sampler.forward)

llm.generate(["Benchmark: "], SamplingParams())
torch.cuda.synchronize()
torch.cuda.profiler.start()
llm.generate(prompts, sps, use_tqdm=False)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("done")
