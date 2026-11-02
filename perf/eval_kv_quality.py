"""Does the KV cache dtype hurt the model? Teacher-forced next-token loss with the KV cache in the loop.

Each text is prefilled in two chunks (the second one reads the first from the cache: the prefix-cache path), then the
rest is fed one true token at a time through the decode path (paged KV cache + CUDA graphs, as the engine runs it).
At every position we record log p(true next token) and the argmax. Run once per KV dtype, then compare:

    python perf/eval_kv_quality.py --out perf/results/quality_bf16.pt
    python perf/eval_kv_quality.py --kv-cache-dtype fp8 --out perf/results/quality_fp8.pt
    python perf/eval_kv_quality.py --compare perf/results/quality_bf16.pt perf/results/quality_fp8.pt

Texts: the 16 longest package descriptions (dist-info METADATA) in the venv, sorted by name: plain English prose and
markdown, available offline.
"""
import argparse
import glob
import os
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--kv-cache-dtype", type=str, default="")
parser.add_argument("--kv-cache-scales", type=str, default="",
                    help="fp8 scales: 'k,v' for all layers, or a JSON file from calibrate_kv_scales.py")
parser.add_argument("--out", type=str)
parser.add_argument("--compare", nargs=2)
parser.add_argument("--num-texts", type=int, default=16)
parser.add_argument("--length", type=int, default=1536, help="tokens per text")
parser.add_argument("--chunk", type=int, default=256, help="first prefill chunk; the second chunk is the same size")
args = parser.parse_args()

if args.compare:
    a, b = (torch.load(f) for f in args.compare)
    for name, r in zip(args.compare, (a, b)):
        print(f"{os.path.basename(name):24s} mean NLL {r['nll'].mean():.4f}   ppl {r['nll'].mean().exp():.3f}")
    d = b["nll"] - a["nll"]
    agree = (a["argmax"] == b["argmax"]).float()
    print(f"NLL difference          mean {d.mean():+.4f}   mean |diff| {d.abs().mean():.4f}   max |diff| {d.abs().max():.3f}")
    print(f"argmax agreement        {agree.mean():.2%} of {agree.numel()} positions")
    n = a["nll"].shape[1]
    for lo in range(0, n, n // 4):
        print(f"  positions {lo:4d}-{lo + n // 4 - 1:4d}: NLL diff {d[:, lo:lo + n // 4].mean():+.4f}, "
              f"argmax agreement {agree[:, lo:lo + n // 4].mean():.2%}")
    sys.exit()

from nanovllm import LLM
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler

extra = dict(kv_cache_dtype=args.kv_cache_dtype) if args.kv_cache_dtype else {}
if args.kv_cache_scales:
    is_pair = "," in args.kv_cache_scales
    extra["kv_cache_scales"] = tuple(map(float, args.kv_cache_scales.split(","))) if is_pair else args.kv_cache_scales
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), enforce_eager=False, max_model_len=4096, **extra)
runner, block_manager = llm.model_runner, llm.scheduler.block_manager

files = glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "*.dist-info", "METADATA"))
files = sorted(files, key=os.path.getsize)[-args.num_texts * 3:]    # calibrate_kv_scales.py never uses these
texts = []
for f in sorted(files):
    ids = llm.tokenizer.encode(open(f, encoding="utf-8", errors="ignore").read())
    if len(ids) >= args.length:
        texts.append(ids[:args.length])
texts = texts[:args.num_texts]
assert len(texts) == args.num_texts, f"only {len(texts)} texts with {args.length} tokens"

logprobs = []
orig_sampler = Sampler.forward
Sampler.forward = lambda self, logits, temperatures: (logprobs.append(logits.float().log_softmax(-1)), orig_sampler(self, logits, temperatures))[1]

prefix = 2 * args.chunk
seqs = [Sequence(t[:prefix]) for t in texts]
for seq in seqs:
    block_manager.allocate(seq, 0)
for start in (0, args.chunk):    # two prefill chunks; the second reads the first one's KV from the cache
    for seq in seqs:
        seq.num_cached_tokens, seq.num_scheduled_tokens = start, args.chunk
    runner.run(seqs, True)

nll, argmax = [], []
for pos in range(prefix, args.length):
    lp = logprobs.pop()
    target = torch.tensor([t[pos] for t in texts], device=lp.device)
    nll.append(-lp.gather(1, target[:, None])[:, 0].cpu())
    argmax.append(lp.argmax(-1).cpu())
    if pos == args.length - 1:
        break
    for seq, t in zip(seqs, texts):
        seq.append_token(t[pos])
        block_manager.may_append(seq)
    runner.run(seqs, False)

result = dict(nll=torch.stack(nll, 1), argmax=torch.stack(argmax, 1))
torch.save(result, args.out)
print(f"{len(texts)} texts x {result['nll'].shape[1]} positions: mean NLL {result['nll'].mean():.4f} -> {args.out}")
