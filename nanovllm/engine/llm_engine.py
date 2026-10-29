# =============================================================================
# llm_engine.py —— 引擎的总入口：把"调度器"和"模型执行器"串成一个循环
#
# 整体伪代码：
#
#   generate(prompts):
#       for 每个 prompt:
#           tokenize → 包装成 Sequence → 放进 scheduler.waiting 队列
#       while 还有没完成的 Sequence:
#           step()                                  # 一个 step = 模型跑一次前向
#       按请求顺序把生成的 token 解码成文字，返回
#
#   step():
#       seqs, is_prefill = scheduler.schedule()     # ① 这一步跑哪些序列？做 prefill 还是 decode？
#       token_ids = model_runner.run(seqs)          # ② 在 GPU 上跑一次前向 + 采样，每个序列得到 1 个新 token
#       scheduler.postprocess(seqs, token_ids)      # ③ 把新 token 追加到序列上，判断是否结束、释放 KV cache
#
# 注意：整个引擎是同步、单线程的。CPU 做完调度 → 交给 GPU 跑 → 等结果回到 CPU（.tolist()）
# → 再调度下一步。CPU 干活的时候 GPU 是空闲的，这是 SGLang "overlap scheduler" 想要消除的开销。
# =============================================================================
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 从 kwargs 里只挑出 Config 认识的字段（比如 enforce_eager、max_model_len），其他的忽略
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # block 大小是类变量：所有 Sequence 共享，用来算"这个序列占几个 block"
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        # Tensor Parallel：除了当前进程（rank 0），再给其他每张 GPU 起一个子进程。
        # 子进程里的 ModelRunner 会进入 loop()，等 rank 0 通过共享内存发来"调用哪个方法"。
        # 只有一张卡时（tensor_parallel_size=1）这个循环不执行。
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 的 ModelRunner 在本进程里。构造时会：加载权重 → 预热测显存 → 分配 KV cache → 捕获 CUDA graph
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        # 注意顺序：Scheduler 必须在 ModelRunner 之后创建，
        # 因为 KV cache 有多少个 block（config.num_kvcache_blocks）是 ModelRunner 量完显存才算出来的
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # prompt 可以是字符串，也可以是已经 tokenize 好的 token id 列表（bench.py 用的是后者）
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)   # 进 waiting 队列，等待 prefill

    def step(self):
        # 整个引擎最核心的三行：调度 → 执行 → 后处理
        seqs, is_prefill = self.scheduler.schedule()
        # num_tokens 只用来在进度条上显示吞吐量：
        #   prefill：这一步处理了多少个 prompt token（正数）
        #   decode ：这一步有多少个序列，每个序列生成 1 个 token（用负数表示"这是 decode"）
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # call() 在 TP 时会先把调用广播给其他 rank，再在本地执行；返回每个序列采样出的新 token id
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 可以所有请求共用一个 SamplingParams，也可以每个请求各给一个（bench.py 每个请求的 max_tokens 不同）
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 离线模式：所有请求一开始就全部加进 waiting 队列（不是"陆续到达"）
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 序列完成的顺序和输入顺序不一样（短的先完成），按 seq_id 排序恢复成输入顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
