# =============================================================================
# trace_engine.py —— 一步一步打印 nano-vllm 的调度过程，用来"亲眼看到"引擎的各个机制
#
# 默认场景（每个机制都会触发一次）：
#   - KV cache 只有 6 个 block（每个 256 个 token）     → 空间不够，触发抢占（preemption）
#   - 每步最多处理 512 个 token                         → A 的 766 个 token 要分两步读（chunked prefill）
#   - A 和 B 的前 512 个 token 完全一样                → B 直接复用 A 的前 2 个 block（prefix caching）
#
# 用法（在 nano-vllm 目录下，先 source ~/venvs/nanovllm/bin/activate）：
#   python tools/trace_engine.py                      # 默认场景
#   python tools/trace_engine.py --blocks 5           # 只有 5 个 block，会发生什么？
#   python tools/trace_engine.py --prefix 600         # 共同前缀 600 个 token，能复用几个 block？
#   python tools/trace_engine.py --budget 2048        # 每步预算 2048，还会 chunked prefill 吗？
#   python tools/trace_engine.py --help               # 所有参数
#
# 建议：每次改参数之前，先在纸上写下你的预测，再运行对照。
#
# 实现：只是给 Scheduler / BlockManager 的几个方法包了一层打印，不改变引擎的行为。
# 为了启动快，用的是 enforce_eager=True（不录 CUDA graph），调度逻辑和正常运行完全一样。
# =============================================================================
import argparse
import os
import random

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler

BLOCK = 256

parser = argparse.ArgumentParser(description="一步一步打印 nano-vllm 的调度过程")
parser.add_argument("--blocks", type=int, default=6, help="KV cache 一共有几个 block（默认 6）")
parser.add_argument("--budget", type=int, default=512, help="每步最多处理多少个 token（max_num_batched_tokens，默认 512）")
parser.add_argument("--prefix", type=int, default=512, help="A 和 B 共同前缀的长度（默认 512）")
parser.add_argument("--a-extra", type=int, default=254, help="A 在共同前缀之后还有多少个 token（默认 254，A 共 766）")
parser.add_argument("--b-extra", type=int, default=100, help="B 在共同前缀之后还有多少个 token（默认 100，B 共 612）")
parser.add_argument("--c-len", type=int, default=400, help="C 的长度，和 A、B 无关（默认 400）")
parser.add_argument("--max-tokens", type=str, default="6,10,10", help="A,B,C 各生成多少个 token（默认 6,10,10）")
parser.add_argument("--model", type=str, default="~/huggingface/Qwen3-0.6B/")
args = parser.parse_args()

# ---- 构造三个请求（token 是随机的，内容不重要，重要的是长度和哪些部分相同）----
random.seed(0)
rand = lambda n: [random.randint(1000, 20000) for _ in range(n)]
P = rand(args.prefix)
A = P + rand(args.a_extra)
B = P + rand(args.b_extra)
C = rand(args.c_len)
max_tokens = [int(x) for x in args.max_tokens.split(",")]
assert len(max_tokens) == 3, "--max-tokens 要给 3 个数，例如 6,10,10"

# 每个请求单独都要能放得下，否则它永远调度不了（引擎会卡住或报错）
for name, prompt, m in zip("ABC", (A, B, C), max_tokens):
    need = -(-(len(prompt) + m) // BLOCK)
    if need > args.blocks:
        raise SystemExit(f"{name} 生成完需要 {need} 个 block，但只有 {args.blocks} 个：单独一个请求都放不下，请调大 --blocks")

names = {}        # seq_id → "A" / "B" / "C"
step_no = [0]
events = []       # 这一步调度过程中发生的事（分配、抢占），在 step 那一行下面打印

llm = LLM(os.path.expanduser(args.model), enforce_eager=True, max_num_batched_tokens=args.budget)
# 换成一个只有 args.blocks 个 block 的管理器（真正的 KV cache 有几百个 block，这里只用前几个）
llm.scheduler.block_manager = BlockManager(args.blocks, BLOCK)

# ---- 给几个方法包一层打印 ----
orig_schedule, orig_post, orig_preempt = Scheduler.schedule, Scheduler.postprocess, Scheduler.preempt
orig_allocate = BlockManager.allocate


def n(seq):
    return names[seq.seq_id]


def allocate(self, seq, num_cached_blocks):
    orig_allocate(self, seq, num_cached_blocks)
    hit = f"，前缀缓存命中 {num_cached_blocks} 个 block = {num_cached_blocks * BLOCK} 个 token" if num_cached_blocks else ""
    events.append(f"分配 {n(seq)}：block_table={seq.block_table}{hit}")


def preempt(self, seq):
    events.append(f"抢占 {n(seq)}（没有空闲 block）：收回它的 block {seq.block_table}，放回 waiting 队头")
    orig_preempt(self, seq)


def schedule(self):
    seqs, is_prefill = orig_schedule(self)
    step_no[0] += 1
    kind = "PREFILL" if is_prefill else "DECODE "
    parts = []
    for s in seqs:
        a, b = s.num_cached_tokens, s.num_cached_tokens + s.num_scheduled_tokens
        what = f"算 token[{a}:{b}]" if is_prefill else f"算 1 个 token（位置 {b - 1}）"
        parts.append(f"{n(s)} {what}")
    print(f"step {step_no[0]:2d} {kind} | " + "；".join(parts))
    for e in events:
        print(f"         · {e}")
    events.clear()
    return seqs, is_prefill


def postprocess(self, seqs, token_ids, is_prefill):
    orig_post(self, seqs, token_ids, is_prefill)
    done = [n(s) for s in seqs if s.is_finished]
    tables = "  ".join(f"{n(s)}={s.block_table}" for s in list(self.running) + list(self.waiting) if s.block_table)
    status = (f"running={[n(s) for s in self.running]} waiting={[n(s) for s in self.waiting]} "
              f"空闲 block={len(self.block_manager.free_block_ids)}  {tables}")
    print(f"         → {status}" + (f"  完成 {done}" if done else ""))


BlockManager.allocate = allocate
Scheduler.schedule, Scheduler.postprocess, Scheduler.preempt = schedule, postprocess, preempt

orig_add = llm.add_request


def add_request(prompt, sp):
    orig_add(prompt, sp)
    seq = llm.scheduler.waiting[-1]
    names[seq.seq_id] = "ABC"[len(names)]


llm.add_request = add_request

sps = [SamplingParams(temperature=0.6, max_tokens=m, ignore_eos=True) for m in max_tokens]
print(f"A = {len(A)} 个 token（生成 {max_tokens[0]} 个）；B = {len(B)} 个 token（生成 {max_tokens[1]} 个，"
      f"前 {args.prefix} 个和 A 相同）；C = {len(C)} 个 token（生成 {max_tokens[2]} 个）")
print(f"KV cache：{args.blocks} 个 block × {BLOCK} token；每步最多 {args.budget} 个 token\n")
llm.generate([A, B, C], sps, use_tqdm=False)
