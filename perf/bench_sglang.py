"""Same workload as nano-vllm's bench.py (seed 0, 256 seqs, in/out 100-1024), run on SGLang's offline Engine."""
import os
import time
from random import randint, seed

import sglang as sgl


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # identical RNG call order to bench.py -> identical prompts and lengths
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    max_tokens = [randint(100, max_ouput_len) for _ in range(num_seqs)]
    sampling_params = [dict(temperature=0.6, ignore_eos=True, max_new_tokens=m) for m in max_tokens]

    tuned = os.environ.get("TUNED") == "1"
    extra = dict(mem_fraction_static=0.85, cuda_graph_max_bs_decode=256) if tuned else {}
    print("engine kwargs:", extra, flush=True)
    llm = sgl.Engine(model_path=path, log_level="error", **extra)
    llm.generate(prompt="Benchmark: ", sampling_params=dict(max_new_tokens=16))   # warmup, like bench.py
    total_tokens = sum(max_tokens)
    for r in range(3):
        t = time.time()
        llm.generate(input_ids=prompt_token_ids, sampling_params=sampling_params)
        t = time.time() - t
        print(f"[sglang run {r + 1}] Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {total_tokens / t:.2f}tok/s", flush=True)
    llm.shutdown()


if __name__ == "__main__":
    main()
