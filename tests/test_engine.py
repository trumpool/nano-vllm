"""End to end through the engine: generation with prefix-cache hits and chunked prefill, and teacher-forced quality.
Each configuration runs in its own process (the engine binds a fixed NCCL port and sizes its KV cache to the whole
GPU). Takes a few minutes; skip with `pytest -m "not slow"`.

Why free-running generations are not compared token by token: "greedy" here is temperature 1e-5 over bf16 logits,
which have exact ties, and the sampler breaks ties randomly. Two runs of the *same* configuration can diverge, and
after one flipped token the rest differs. Teacher forcing (perf/eval_kv_quality.py) feeds every configuration the
same tokens and compares per-position log-probs instead.
"""
import functools
import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.expanduser(os.environ.get("NANOVLLM_MODEL", "~/huggingface/Qwen3-0.6B/"))
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not os.path.isdir(MODEL), reason=f"needs the model at {MODEL} (or set NANOVLLM_MODEL)"),
]
ENV = dict(os.environ, PYTHONPATH=ROOT)
# per prompt of engine_run.py: something a correct continuation contains (None: Qwen3-0.6B gets it wrong anyway)
EXPECTED = ["Paris", "Au", "100", "red", "return fibonacci(n-1)", None, "Bonjour", "2, 3, 5, 7, 11", "block", "BlockManager"]


@functools.cache
def generate(**config):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "engine_run.py"), MODEL, json.dumps(config)],
                         capture_output=True, text=True, env=ENV)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_fp8_doubles_kv_capacity():
    bf16, fp8 = generate(), generate(kv_cache_dtype="fp8")
    assert fp8["num_kvcache_blocks"] >= 1.9 * bf16["num_kvcache_blocks"]


def test_fp8_prefix_cache_and_chunked_prefill_use_flashinfer():
    assert generate()["prefill_plans"] == 0
    assert generate(kv_cache_dtype="fp8")["prefill_plans"] >= 2    # chunk 2 of the document in round 1, round 2


@pytest.mark.parametrize("config", [{}, dict(kv_cache_dtype="fp8"), dict(kv_cache_dtype="fp8", enforce_eager=True)],
                         ids=["bf16", "fp8-cudagraph", "fp8-eager"])
def test_generations_are_correct(config):
    texts = generate(**config)["texts"]
    assert len(texts) == len(EXPECTED)
    for text, expected in zip(texts, EXPECTED):
        assert expected is None or expected in text, text


@functools.cache
def teacher_forced(num_texts, *args):
    path = os.path.join(ROOT, "perf", "results", f"test_quality_{abs(hash(args))}_{num_texts}.pt")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "perf", "eval_kv_quality.py"), "--num-texts", str(num_texts),
                          "--length", "1024", "--out", path, *args], capture_output=True, text=True, env=ENV)
    assert out.returncode == 0, out.stderr[-3000:]
    result = torch.load(path)
    os.remove(path)
    return result


def mean_abs_nll_diff(a, b):
    return (a["nll"] - b["nll"]).abs().mean().item()


def test_teacher_forced_quality():
    bf16, fp8 = teacher_forced(16), teacher_forced(16, "--kv-cache-dtype", "fp8")
    assert (fp8["nll"] - bf16["nll"]).mean().abs() < 0.05
    assert (fp8["argmax"] == bf16["argmax"]).float().mean() > 0.9


def test_fp8_cuda_graphs_match_eager():
    # 16 sequences: the batch exactly fills a captured graph, the graph replays the eager plan: bit-identical
    graph, eager = teacher_forced(16, "--kv-cache-dtype", "fp8"), teacher_forced(16, "--kv-cache-dtype", "fp8", "--enforce-eager")
    assert torch.equal(graph["nll"], eager["nll"]) and torch.equal(graph["argmax"], eager["argmax"])
    # 5 sequences: the batch-size-8 graph, with 3 padding requests. FlashInfer then splits the KV work differently
    # (so does flash-attn for bf16), which changes the summation order: not bit-identical, but well below the fp8 effect
    graph, eager = teacher_forced(5, "--kv-cache-dtype", "fp8"), teacher_forced(5, "--kv-cache-dtype", "fp8", "--enforce-eager")
    bf16 = teacher_forced(5, "--enforce-eager")
    assert mean_abs_nll_diff(graph, eager) < 0.75 * mean_abs_nll_diff(bf16, eager)
    assert (graph["argmax"] == eager["argmax"]).float().mean() > 0.97
