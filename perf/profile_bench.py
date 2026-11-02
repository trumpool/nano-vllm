"""Profile nano-vllm on the bench.py workload: where does the time go, and how often does the KV cache run out?

Same workload as bench.py (256 seqs, input/output lengths uniform in [100, 1024], seed 0, same RNG call order).
The engine is instrumented by wrapping a few methods; the engine code itself is unchanged.

    python perf/profile_bench.py                    # real KV capacity
    python perf/profile_bench.py --blocks 200       # pretend the KV cache only has 200 blocks
    python perf/profile_bench.py --json steps.json  # also dump one record per engine step

GPU time is measured with CUDA events around run_model() and the sampler, so it is the GPU-side window of a
step (including any gaps while the CPU is still launching kernels). Everything else in a step is CPU-only:
nano-vllm syncs every step (sampler output .tolist()), so the GPU is idle while the CPU schedules, prepares
inputs and postprocesses.
"""
import argparse
import json
import os
from random import randint, seed
from time import perf_counter

import numpy as np
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.layers.sampler import Sampler

parser = argparse.ArgumentParser()
parser.add_argument("--blocks", type=int, default=0, help="use only this many KV blocks (0 = all)")
parser.add_argument("--num-seqs", type=int, default=256)
parser.add_argument("--json", type=str, default="", help="dump per-step records to this file")
parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
parser.add_argument("--kv-cache-dtype", type=str, default="", help="passed to LLM() if set (e.g. fp8)")
args = parser.parse_args()

# ---- workload: identical to bench.py ----
seed(0)
extra = dict(kv_cache_dtype=args.kv_cache_dtype) if args.kv_cache_dtype else {}
llm = LLM(os.path.expanduser(args.model), enforce_eager=False, max_model_len=4096, **extra)
prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(args.num_seqs)]
sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024)) for _ in range(args.num_seqs)]

block_size = llm.scheduler.block_size
real_blocks = len(llm.scheduler.block_manager.blocks)
if args.blocks:
    assert args.blocks <= real_blocks, f"only {real_blocks} blocks exist"
    llm.scheduler.block_manager = BlockManager(args.blocks, block_size)
num_blocks = len(llm.scheduler.block_manager.blocks)

# ---- instrumentation ----
stats = {}
steps = []
cur = {}
preempted = set()


def reset():
    stats.update(preempts=0, preempted_kv_tokens=0, recompute_tokens=0, recompute_hit_tokens=0)
    steps.clear()
    preempted.clear()


reset()
o_step, o_schedule, o_postprocess, o_preempt = LLMEngine.step, Scheduler.schedule, Scheduler.postprocess, Scheduler.preempt
o_allocate = BlockManager.allocate
o_prep_prefill, o_prep_decode, o_run_model = ModelRunner.prepare_prefill, ModelRunner.prepare_decode, ModelRunner.run_model
o_sampler = Sampler.forward


def timed_cpu(key, fn):
    def wrapper(*a, **kw):
        t = perf_counter()
        out = fn(*a, **kw)
        cur[key] = cur.get(key, 0.0) + perf_counter() - t
        return out
    return wrapper


def timed_gpu(fn):
    def wrapper(*a, **kw):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*a, **kw)
        end.record()
        cur["gpu_events"].append((start, end))
        return out
    return wrapper


def schedule(self):
    seqs, is_prefill = o_schedule(self)
    cur["num_seqs"] = len(seqs)
    cur["kv_tokens"] = sum(len(s) for s in seqs)   # context the attention has to read this step
    if is_prefill:
        # preemption only hits running seqs, whose prompt was already computed once: every prefill token of a
        # previously preempted seq is recomputation
        stats["recompute_tokens"] += sum(s.num_scheduled_tokens for s in seqs if s.seq_id in preempted)
        ids = {s.seq_id for s in seqs}
        cur["stalled"] = sum(1 for s in self.running if s.seq_id not in ids)   # running seqs that don't advance
    return seqs, is_prefill


def preempt(self, seq):
    stats["preempts"] += 1
    stats["preempted_kv_tokens"] += seq.num_cached_tokens
    preempted.add(seq.seq_id)
    o_preempt(self, seq)


def allocate(self, seq, num_cached_blocks):
    o_allocate(self, seq, num_cached_blocks)
    if seq.seq_id in preempted:
        stats["recompute_hit_tokens"] += num_cached_blocks * self.block_size


def step(self):
    cur.clear()
    cur["gpu_events"] = []
    t = perf_counter()
    outputs, num_tokens = o_step(self)
    wall = perf_counter() - t
    gpu = sum(a.elapsed_time(b) for a, b in cur["gpu_events"]) / 1e3   # already synced by .tolist()
    steps.append(dict(
        kind="prefill" if num_tokens > 0 else "decode",
        num_seqs=cur["num_seqs"],
        num_tokens=abs(num_tokens),
        kv_tokens=cur["kv_tokens"],
        wall=wall,
        gpu=gpu,
        schedule=cur.get("schedule", 0.0),
        prepare=cur.get("prepare", 0.0),
        postprocess=cur.get("postprocess", 0.0),
        used_blocks=len(self.scheduler.block_manager.used_block_ids),
        running=len(self.scheduler.running),
        waiting=len(self.scheduler.waiting),
        stalled=cur.get("stalled", 0),
    ))
    return outputs, num_tokens


LLMEngine.step = step
Scheduler.schedule = timed_cpu("schedule", schedule)
Scheduler.postprocess = timed_cpu("postprocess", o_postprocess)
Scheduler.preempt = preempt
BlockManager.allocate = allocate
ModelRunner.prepare_prefill = timed_cpu("prepare", o_prep_prefill)
ModelRunner.prepare_decode = timed_cpu("prepare", o_prep_decode)
ModelRunner.run_model = timed_gpu(o_run_model)
Sampler.forward = timed_gpu(o_sampler)

# ---- run ----
llm.generate(["Benchmark: "], SamplingParams())
reset()
t = perf_counter()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
total_time = perf_counter() - t

# ---- report ----
out_tokens = sum(sp.max_tokens for sp in sampling_params)
prompt_tokens = sum(map(len, prompt_token_ids))
need_blocks = sum(-(-(len(p) + sp.max_tokens) // block_size) for p, sp in zip(prompt_token_ids, sampling_params))
pre = [s for s in steps if s["kind"] == "prefill"]
dec = [s for s in steps if s["kind"] == "decode"]
tot = lambda xs, k: sum(x[k] for x in xs)
pct = lambda a, b: f"{100 * a / b:5.1f}%"

print(f"\n=== nano-vllm profile: {args.num_seqs} seqs, {prompt_tokens} prompt + {out_tokens} output tokens ===")
print(f"throughput        {out_tokens / total_time:8.1f} tok/s   ({total_time:.2f} s; bench.py baseline 4543 tok/s)")
print(f"KV cache          {num_blocks} blocks x {block_size} = {num_blocks * block_size} tokens "
      f"(all seqs at once would need {need_blocks} blocks)")

wall = tot(steps, "wall")
print(f"\n--- where the time goes ({len(steps)} steps, {wall:.2f} s inside step()) ---")
for name, xs in (("prefill", pre), ("decode", dec)):
    w = tot(xs, "wall")
    print(f"{name:8s} steps {len(xs):5d}  time {w:6.2f} s {pct(w, wall)}  | GPU {pct(tot(xs, 'gpu'), w)}  "
          f"schedule {pct(tot(xs, 'schedule'), w)}  prepare {pct(tot(xs, 'prepare'), w)}  "
          f"postprocess {pct(tot(xs, 'postprocess'), w)}")
gpu = tot(steps, "gpu")
print(f"all      GPU busy {gpu:.2f} s {pct(gpu, wall)};  GPU idle (CPU-only work) {wall - gpu:.2f} s {pct(wall - gpu, wall)}")

bs = np.array([s["num_seqs"] for s in dec])
print(f"\n--- decode batch size ---")
print(f"mean {bs.mean():.1f}   p10 {np.percentile(bs, 10):.0f}   p50 {np.percentile(bs, 50):.0f}   "
      f"p90 {np.percentile(bs, 90):.0f}   max {bs.max()}")
print(f"{'bs range':>10s} {'steps':>6s} {'tokens':>7s} {'ms/step':>8s} {'tok/s':>8s}")
for lo, hi in ((1, 8), (9, 32), (33, 64), (65, 96), (97, 128), (129, 192), (193, 256), (257, 512)):
    xs = [s for s in dec if lo <= s["num_seqs"] <= hi]
    if xs:
        n, w = tot(xs, "num_seqs"), tot(xs, "wall")
        print(f"{lo:4d}-{hi:<5d} {len(xs):6d} {n:7d} {1e3 * w / len(xs):8.2f} {n / w:8.0f}")

# decode step time ~ fixed cost (weights, launch) + per-seq cost + per-context-token cost (reading the KV cache)
X = np.array([[1, s["num_seqs"], s["kv_tokens"]] for s in dec], dtype=float)
y = np.array([s["gpu"] for s in dec])
(c0, c_seq, c_kv), *_ = np.linalg.lstsq(X, y, rcond=None)
hf = llm.model_runner.config.hf_config
kv_bytes = 2 * hf.num_hidden_layers * hf.num_key_value_heads * hf.head_dim * llm.model_runner.kv_cache.element_size()
kv_share = c_kv * X[:, 2].sum() / y.sum()
print(f"fit (GPU time): {1e3 * c0:.2f} ms + {1e6 * c_seq:.1f} us/seq + {1e9 * c_kv:.2f} ns/context token "
      f"-> KV read at {kv_bytes / c_kv / 1e9:.0f} GB/s; KV reads = {kv_share:.0%} of decode GPU time")

print(f"\n--- KV capacity ---")
print(f"preemptions             {stats['preempts']} ({len(preempted)} distinct seqs)")
print(f"KV tokens thrown away   {stats['preempted_kv_tokens']}")
print(f"recomputed prefill      {stats['recompute_tokens']} tokens "
      f"(+{stats['recompute_hit_tokens']} recovered from prefix cache) = "
      f"{pct(stats['recompute_tokens'], tot(pre, 'num_tokens'))} of all prefill tokens")
print(f"KV blocks in use        mean {np.mean([s['used_blocks'] for s in dec]) / num_blocks:.1%} during decode")
stalled = tot(pre, "stalled")
print(f"decode stalled by prefill  {stalled} seq-steps (running seqs that did not advance during a prefill step)")

if args.json:
    with open(args.json, "w") as f:
        json.dump(dict(args=vars(args), num_blocks=num_blocks, total_time=total_time, stats=stats, steps=steps), f)
    print(f"\nper-step records -> {args.json}")
