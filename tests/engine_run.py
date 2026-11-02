"""Helper for test_engine.py: run one engine configuration and print the outputs as JSON (last stdout line).

Round 1 generates for short factual prompts plus a ~1500-token document, which is longer than
max_num_batched_tokens and so is prefilled in chunks (the second chunk reads the first from the cache). Round 2 asks
another question about the same document: its full blocks are prefix-cache hits from round 1.
"""
import glob
import json
import os
import sys

from nanovllm import LLM, SamplingParams
import nanovllm.engine.model_runner as model_runner

model, kwargs = sys.argv[1], json.loads(sys.argv[2])
llm = LLM(model, max_model_len=2048, max_num_batched_tokens=1024, **kwargs)

prefill_plans = 0
plan_attention = model_runner.plan_attention


def counting_plan_attention(wrapper, *args, **kwargs):
    global prefill_plans
    prefill_plans += type(wrapper).__name__ == "BatchPrefillWithPagedKVCacheWrapper"
    return plan_attention(wrapper, *args, **kwargs)


model_runner.plan_attention = counting_plan_attention

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(root, "nanovllm", "**", "*.py"), recursive=True)))
document = llm.tokenizer.decode(llm.tokenizer.encode(source)[:1500])
prompts = [
    "The capital of France is",
    "The chemical symbol for gold is",
    "Water boils at a temperature of",
    "The three primary colors are",
    "def fibonacci(n):",
    "The largest planet in the solar system is",
    "Translate to French: 'Good morning, how are you?'",
    "List the first ten prime numbers:",
]
params = SamplingParams(temperature=1e-5, max_tokens=48)    # greedy (the sampler does not allow temperature 0)
round1 = llm.generate(prompts + [document + "\n\nQuestion: what does the scheduler do? Answer:"], params, use_tqdm=False)
round2 = llm.generate([document + "\n\nQuestion: how is the KV cache allocated? Answer:"], params, use_tqdm=False)
print(json.dumps(dict(
    outputs=[o["token_ids"] for o in round1 + round2],
    texts=[o["text"] for o in round1 + round2],
    prefill_plans=prefill_plans,
    num_kvcache_blocks=llm.model_runner.config.num_kvcache_blocks,
)))
