"""Throughput vs number of concurrent requests, bf16 vs fp8 KV cache.

Requests have bench.py's length distribution (input and output uniform in [100, 1024], ignore_eos); the first n of a
fixed list of 512 (seed 0) are submitted at once and run to completion. Small n: the KV cache is not the limit, and
fp8 can only win on attention bandwidth. Large n: bf16 runs out of KV blocks (smaller decode batches, preemption and
recomputation) and fp8's doubled capacity counts too. One engine (one KV dtype) per process:

    python perf/sweep.py --out perf/results/sweep_bf16.json
    python perf/sweep.py --kv-cache-dtype fp8 --out perf/results/sweep_fp8.json
    python perf/sweep.py --plot perf/results/sweep_*.json --png perf/results/sweep.png

Repeated runs of the same dtype (several --out files) are averaged by --plot.
"""
import argparse
import json
import os
from collections import defaultdict
from random import randint, seed
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument("--kv-cache-dtype", type=str, default="auto")
parser.add_argument("--num-seqs", type=str, default="8,16,32,64,128,256,512")
parser.add_argument("--out", type=str)
parser.add_argument("--plot", nargs="+")
parser.add_argument("--png", type=str, default="perf/results/sweep.png")
args = parser.parse_args()


def plot(files, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = defaultdict(list)
    for f in files:
        r = json.load(open(f))
        runs[r["kv_cache_dtype"]].append({p["num_seqs"]: p for p in r["points"]})
    mean = lambda dtype, n, key: sum(r[n][key] for r in runs[dtype]) / len(runs[dtype])
    ns = sorted(runs["auto"][0])
    tput = {d: [mean(d, n, "throughput") for n in ns] for d in ("auto", "fp8")}

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    style = {"auto": ("#2a78d6", "o", "bf16 KV cache"), "fp8": ("#eb6834", "s", "fp8 KV cache")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=surface, gridspec_kw=dict(width_ratios=[3, 2]))
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9, length=0)
        ax.set_xlabel("concurrent requests", color=ink2, fontsize=9)
    for d in ("auto", "fp8"):
        color, marker, label = style[d]
        ax1.plot(ns, tput[d], color=color, marker=marker, markersize=6, linewidth=2, label=label,
                 markeredgecolor=surface, markeredgewidth=1.5)
        ax1.annotate(label.split()[0], (ns[-1], tput[d][-1]), xytext=(8, 0), textcoords="offset points",
                     va="center", color=ink, fontsize=9)
    ax1.set_ylim(0, max(tput["fp8"]) * 1.12)
    ax1.set_title("Output throughput (tok/s)", loc="left", color=ink, fontsize=10)
    ax1.legend(frameon=False, fontsize=9, labelcolor=ink2, loc="upper left")
    speedup = [b / a for a, b in zip(tput["auto"], tput["fp8"])]
    ax2.axhline(1, color=ink2, linewidth=0.8, linestyle=(0, (2, 2)))
    ax2.plot(ns, speedup, color=ink, marker="D", markersize=5, linewidth=2, markeredgecolor=surface)
    for i in (0, speedup.index(max(speedup))):    # smallest batch and the peak
        ax2.annotate(f"{speedup[i]:.2f}x", (ns[i], speedup[i]), xytext=(0, 9), textcoords="offset points",
                     ha="center", color=ink, fontsize=9)
    ax2.set_ylim(0.8, max(speedup) * 1.15)
    ax2.set_title("fp8 / bf16", loc="left", color=ink, fontsize=10)
    fig.suptitle("nano-vllm, Qwen3-0.6B, RTX 5070 Ti: input/output lengths uniform in [100, 1024]",
                 x=0.01, ha="left", color=ink2, fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=150, facecolor=surface)
    print(f"-> {png}")
    print(f"{'n':>5s} " + " ".join(f"{d + ' tok/s':>12s} {d + ' pre':>8s} {d + ' bs':>8s}" for d in ("auto", "fp8")) + "   speedup")
    for i, n in enumerate(ns):
        cols = " ".join(f"{mean(d, n, 'throughput'):12.0f} {mean(d, n, 'preemptions'):8.0f} {mean(d, n, 'mean_decode_bs'):8.1f}"
                        for d in ("auto", "fp8"))
        print(f"{n:5d} {cols}   {speedup[i]:.2f}x")


if args.plot:
    plot(args.plot, args.png)
    raise SystemExit

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler

seed(0)
requests = [([randint(0, 10000) for _ in range(randint(100, 1024))], randint(100, 1024)) for _ in range(512)]
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), max_model_len=4096, kv_cache_dtype=args.kv_cache_dtype)

counts = defaultdict(int)
preempt, step = Scheduler.preempt, LLMEngine.step


def counting_preempt(self, seq):
    counts["preemptions"] += 1
    preempt(self, seq)


def counting_step(self):
    outputs, num_tokens = step(self)
    if num_tokens < 0:
        counts["decode_steps"] += 1
        counts["decode_seqs"] += -num_tokens
    return outputs, num_tokens


Scheduler.preempt, LLMEngine.step = counting_preempt, counting_step
llm.generate(["Benchmark: "], SamplingParams())
points = []
for n in map(int, args.num_seqs.split(",")):
    prompts = [p for p, _ in requests[:n]]
    params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=m) for _, m in requests[:n]]
    counts.clear()
    t = perf_counter()
    llm.generate(prompts, params, use_tqdm=False)
    t = perf_counter() - t
    tokens = sum(m for _, m in requests[:n])
    points.append(dict(num_seqs=n, throughput=tokens / t, time=t, preemptions=counts["preemptions"],
                       mean_decode_bs=counts["decode_seqs"] / counts["decode_steps"]))
    print(f"{args.kv_cache_dtype:5s} n={n:4d}  {tokens / t:8.1f} tok/s  ({t:.2f} s)  preemptions {counts['preemptions']:4d}  "
          f"mean decode batch {points[-1]['mean_decode_bs']:.1f}", flush=True)
with open(args.out, "w") as f:
    json.dump(dict(kv_cache_dtype=args.kv_cache_dtype, num_kvcache_blocks=llm.model_runner.config.num_kvcache_blocks,
                   points=points), f, indent=1)
