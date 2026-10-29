# =============================================================================
# sequence.py —— 一个请求在引擎里的全部状态
#
# 一个 Sequence 记录：
#   token_ids            prompt + 已经生成的所有 token
#   num_prompt_tokens    prompt 有多长（之后的都是生成出来的）
#   num_cached_tokens    已经算过、KV 已经写进 KV cache 的 token 数
#   num_scheduled_tokens 这一步要算多少个 token（prefill 时可能是一整段，decode 时是 1）
#   block_table          这个序列的 KV 存在哪些 block 里（"页表"），比如 [5, 2, 7]
#   status               WAITING（等 prefill）→ RUNNING（decode 中）→ FINISHED
#
# 几个关键关系（block_size = 256）：
#   第 i 个 token 在第 i // 256 个 block 里，块内偏移 i % 256
#   num_blocks = ceil(num_tokens / 256)
#   "未计算的 token" = token_ids[num_cached_tokens:]
# =============================================================================
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()    # 在 waiting 队列：还没 prefill，或者 prefill 只做了一部分，或者被抢占了
    RUNNING = auto()    # 在 running 队列：prompt 已经全部算完，每一步生成 1 个 token
    FINISHED = auto()   # 生成了 EOS 或者达到 max_tokens


class Sequence:
    block_size = 256    # 会被 LLMEngine 用 config.kvcache_block_size 覆盖
    counter = count()   # 全局自增 id：0, 1, 2, ...

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]      # decode 时模型的输入就是"上一个 token"
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 向上取整：1000 个 token → 4 个 block（最后一个 block 只用了 232 个位置）
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个 block 里实际用了几个位置（1 ~ 256）
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 第 i 个 block 对应的 token（用来算 prefix cache 的 hash）
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # ---- 下面两个方法只在 Tensor Parallel 时用到 ----
    # rank 0 要把 Sequence pickle 之后通过共享内存发给其他 rank。
    # 为了少传数据：prefill 时传全部 token（要算整段），decode 时只传最后一个 token（只算它）。
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
