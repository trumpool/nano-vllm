"""Per-layer fp8 KV cache scales: the largest |k| and |v| each layer writes to the cache over a set of calibration
texts, divided by e4m3's max (448), times a margin for values past what calibration saw (they saturate).

    python perf/calibrate_kv_scales.py --out perf/results/kv_scales_qwen3_0.6b.json
    python perf/eval_kv_quality.py --kv-cache-dtype fp8 --kv-cache-scales perf/results/kv_scales_qwen3_0.6b.json \
        --out perf/results/q_fp8_calib.pt

Calibration texts: package descriptions (dist-info METADATA) in the venv, skipping the largest ones, which are the
pool eval_kv_quality.py picks its evaluation texts from.
"""
import argparse
import glob
import json
import os
import sys

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.layers.attention import Attention

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num-texts", type=int, default=64)
parser.add_argument("--length", type=int, default=1024, help="max tokens per text")
parser.add_argument("--margin", type=float, default=1.25)
parser.add_argument("--skip-largest", type=int, default=48, help="eval_kv_quality.py's pool (3 x its --num-texts)")
args = parser.parse_args()

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), enforce_eager=True, max_model_len=4096)

files = glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "*.dist-info", "METADATA"))
files = sorted(files, key=os.path.getsize)[:-args.skip_largest]
texts = []
for f in reversed(files):
    ids = llm.tokenizer.encode(open(f, encoding="utf-8", errors="ignore").read())
    if len(ids) >= 256:
        texts.append(ids[:args.length])
    if len(texts) == args.num_texts:
        break

# record what every layer writes to the cache (k after RoPE, as stored)
layers = [m for m in llm.model_runner.model.modules() if isinstance(m, Attention)]
amax = {id(m): torch.zeros(2, device="cuda") for m in layers}
forward = Attention.forward


def recording_forward(self, q, k, v):
    torch.maximum(amax[id(self)], torch.stack([k.abs().amax().float(), v.abs().amax().float()]), out=amax[id(self)])
    return forward(self, q, k, v)


Attention.forward = recording_forward
llm.generate(texts, SamplingParams(max_tokens=1), use_tqdm=False)
Attention.forward = forward

amax = torch.stack([amax[id(m)] for m in layers]).cpu()
scales = (amax * args.margin / 448).tolist()
result = dict(k_scale=[s[0] for s in scales], v_scale=[s[1] for s in scales],
              k_amax=amax[:, 0].tolist(), v_amax=amax[:, 1].tolist(),
              num_texts=len(texts), num_tokens=sum(map(len, texts)), margin=args.margin)
with open(args.out, "w") as f:
    json.dump(result, f, indent=1)
print(f"{len(texts)} texts, {result['num_tokens']} tokens -> {args.out}")
print(f"{'layer':>5s} {'k amax':>8s} {'v amax':>8s} {'k scale':>8s} {'v scale':>8s}")
for i, (a, s) in enumerate(zip(amax.tolist(), scales)):
    print(f"{i:5d} {a[0]:8.1f} {a[1]:8.1f} {s[0]:8.4f} {s[1]:8.4f}")
