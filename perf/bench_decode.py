"""Decode microbenchmark: how much of a decode step is spent reading the KV cache?

For a grid of (batch size, context length), with every sequence at the same context length:
  attn  one layer of paged decode attention (flash_attn_with_kvcache, bf16), x num_layers
  step  a full decode step through ModelRunner.run() with CUDA graphs, as the engine runs it
  share attn / step

Block tables walk the real KV cache in a fixed random order, wrapping around when bs x blocks/seq exceeds the
cache. A block is only reused after every other block has been, i.e. ~num_blocks MB of reads later per layer,
so the 48 MB L2 can't serve repeated reads (drawing blocks independently at random inflated "bandwidth" to
1.2 TB/s, above the 896 GB/s peak).

    python perf/bench_decode.py
"""
import os
import random
from time import perf_counter

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm import LLM
from nanovllm.engine.sequence import Sequence

BATCH_SIZES = [16, 32, 64, 128, 256]
CONTEXT_LENS = [256, 512, 1024, 2048]


def time_gpu(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def time_wall(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    t = perf_counter()
    for _ in range(iters):
        fn()
    return (perf_counter() - t) / iters


llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), enforce_eager=False, max_model_len=4096)
runner = llm.model_runner
hf = runner.config.hf_config
block_size = runner.block_size
num_blocks = runner.config.num_kvcache_blocks
num_layers, num_heads, num_kv_heads, head_dim = hf.num_hidden_layers, hf.num_attention_heads, hf.num_key_value_heads, hf.head_dim
k_cache, v_cache = runner.kv_cache[0, 0], runner.kv_cache[1, 0]   # layer 0: [num_blocks, block_size, kv_heads, head_dim]
kv_bytes_per_token = 2 * num_kv_heads * head_dim * k_cache.element_size()   # one layer, K + V
random.seed(0)
order = random.sample(range(num_blocks), num_blocks)

print(f"{num_blocks} KV blocks; {kv_bytes_per_token * num_layers / 1024:.0f} KiB of KV per token over {num_layers} layers")
print(f"{'bs':>4s} {'ctx':>5s} | {'attn/layer':>10s} {'BW':>9s} | {'attn x L':>9s} {'step':>8s} {'share':>6s} | {'tok/s':>7s}")
for bs in BATCH_SIZES:
    for ctx in CONTEXT_LENS:
        n = (ctx + block_size - 1) // block_size
        tables = [[order[(i * n + j) % num_blocks] for j in range(n)] for i in range(bs)]

        # attention only, one layer
        q = torch.randn(bs, 1, num_heads, head_dim, dtype=k_cache.dtype, device="cuda")
        block_table = torch.tensor(tables, dtype=torch.int32, device="cuda")
        seqlens = torch.full((bs,), ctx, dtype=torch.int32, device="cuda")
        attn = time_gpu(lambda: flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=seqlens, block_table=block_table,
                                                        softmax_scale=head_dim ** -0.5, causal=True))
        bw = bs * ctx * kv_bytes_per_token / attn / 1e9

        # full decode step (prepare inputs, CUDA graph replay, sampling), as the engine does it
        seqs = []
        for table in tables:
            seq = Sequence([random.randrange(10000) for _ in range(ctx)])
            seq.block_table = table
            seqs.append(seq)
        step = time_wall(lambda: runner.run(seqs, False))

        share = attn * num_layers / step
        wraps = "*" if bs * n > num_blocks else " "
        print(f"{bs:4d} {ctx:5d} | {attn * 1e6:8.1f}us {bw:6.0f}GB/s{wraps}| {attn * num_layers * 1e3:7.2f}ms {step * 1e3:6.2f}ms "
              f"{share:6.0%} | {bs / step:7.0f}")
print("* more block references than blocks: sequences share blocks, and concurrently running ones can hit in L2, "
      "so time is underestimated")
