# =============================================================================
# scheduler.py —— 决定每一步跑哪些序列（continuous batching 的核心）
#
# 两个队列：
#   waiting：等待 prefill 的序列（新请求、prefill 只做了一半的、被抢占回来的）
#   running：prompt 已经算完、正在逐个生成 token 的序列
#
# schedule() 伪代码：
#
#   # 规则 1：prefill 优先。只要能做 prefill，这一步就只做 prefill
#   budget = max_num_batched_tokens                 # 默认 16384 个 token
#   for seq in waiting（从队头开始）:
#       if 是新序列: 查 prefix cache → 分配 KV cache 的 block（不够就停）
#       要算的 token = 还没算的那部分
#       if 预算不够 且 这不是本批第一个序列: 停           # 规则 2：只有第一个序列能被切开（chunked prefill）
#       本步算 min(要算的, 剩余预算) 个 token
#       if 这个序列的 prompt 这一步就能算完: 移到 running
#   if 选到了序列: return (这些序列, is_prefill=True)
#
#   # 规则 3：没有 prefill 可做时，才做 decode
#   for seq in running:
#       while 下一个 token 需要新 block 但没有空闲 block:
#           抢占 running 队尾的序列（释放它的全部 block，扔回 waiting 队头）
#       本步算 1 个 token
#   return (这些序列, is_prefill=False)
#
# 要点：prefill 和 decode 永远不在同一步里混着跑。只要 waiting 里有能做的请求，
# 所有正在 decode 的序列这一步都要停下来等。
# =============================================================================
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                    # 一步最多多少个序列（默认 512）
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 一步最多多少个 token（默认 16384）
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---------------- prefill ----------------
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens   # 本步还剩多少 token 预算
            if remaining == 0:
                break
            if not seq.block_table:
                # 新序列（或被抢占后 block 全释放了的序列）：
                # 先看前缀能命中多少个已缓存的 block，再确认空闲 block 够不够给它全部分配
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:      # 空闲 block 不够 → 这一步不再接新的 prefill
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size   # 命中的前缀不用再算
            else:
                # 已经分配过 block、prefill 做了一部分的序列（上一步被 chunked 了）：接着算剩下的
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                # 预算不够算完这个序列：如果它不是本批第一个，就不切它，留到下一步
                break
            if not seq.block_table:
                # 一次性给整个序列（当前所有 token）分配 block，哪怕这一步只算其中一部分
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 这一步之后 prompt 就算完了 → 移进 running，下一步开始 decode
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 否则（被 chunk 了）序列留在 waiting 队头，下一步继续 prefill
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # ---------------- decode ----------------
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 这个序列的下一个 token 如果正好要开一个新 block，就需要 1 个空闲 block
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 抢占 running 队尾（最晚加入的）序列，腾出它的 block 给当前序列
                    self.preempt(self.running.pop())
                else:
                    # 没有别人可抢了：只能把当前序列自己抢占掉
                    self.preempt(seq)
                    break
            else:
                # while ... else：只有 while 没被 break 时才执行，即"空间够了"
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)    # 需要的话真正分配那个新 block
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把这一步选中的序列按原顺序放回 running 队头（没被选中的仍在后面）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # recompute 式抢占：直接释放这个序列的所有 block，之后重新 prefill（prompt + 已生成的 token 一起算）
        # 不过 block 释放时 hash 还留着，如果这些 block 还没被别人覆盖，重新 prefill 时能命中 prefix cache
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)     # 放到 waiting 队头，优先恢复

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # 这一步算完之后，可能有 block 被写满了 → 给写满的 block 算 hash，登记进 prefix cache
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # prefill 只做了一部分（chunked）：这一步采样出的 token 没有意义，丢掉
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
