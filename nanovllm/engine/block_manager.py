# =============================================================================
# block_manager.py —— KV cache 的"内存分配器"，也就是 PagedAttention 的页表管理 + prefix caching
#
# KV cache 被切成 num_blocks 个固定大小的 block（每个 block 存 256 个 token 的 K 和 V）。
# 每个序列有一个 block_table（页表），记录它的第 0、1、2…个逻辑 block 存在哪个物理 block 里。
# 物理 block 不需要连续，所以不会有"显存碎片"的问题（这就是 PagedAttention 的思路）。
#
# prefix caching（前缀缓存）：
#   每个写满的 block 都有一个 hash = H(前一个 block 的 hash, 本 block 的 256 个 token)
#   因为是链式的，hash 相同 ⇔ 从序列开头到这个 block 结束的整段前缀都相同
#   新序列进来时，从第 0 个 block 开始逐个查 hash 表，命中的 block 直接复用，不用重算
#
#   例子：A 和 B 共享前 600 个 token 的 system prompt
#     block 0 (token 0–255)、block 1 (256–511) 完全相同 → B 可以直接复用 A 的这两个 block
#     block 2 (512–767) 里 600 之后的 token 不同 → hash 不同，B 要自己算
#     所以只能复用 512 个 token，不是 600 个（复用的粒度是整个 block）
#
# 引用计数：一个 block 可能同时被多个序列共享（共同前缀），ref_count = 正在用它的序列数。
# ref_count 降到 0 时 block 回到空闲队列，但 hash 先不删——之后有相同前缀的请求来，还能把它"捞回来"。
# 只有它被真正分配给别的内容时，旧 hash 才会被清掉。
# =============================================================================
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0      # 有几个序列在用这个 block
        self.hash = -1          # -1 = 没写满 / 没登记进 prefix cache
        self.token_ids = []     # 这个 block 存的是哪些 token（hash 命中后再比对一次，防止 hash 冲突）

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # 重新分配给新内容：清掉旧的 hash，引用计数设为 1（分配它的那个序列）
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()     # prefix cache 的索引：hash → 物理 block
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 链式 hash：把"前一个 block 的 hash"也混进去，这样 hash 代表的是整段前缀
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        # 从空闲队列队头取一个 block。队头是最早被释放的，相当于 LRU：最久没用的缓存最先被覆盖
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 这个 block 以前可能缓存过某个前缀，现在要被覆盖了 → 从 hash 表里删掉它的旧记录
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        # 放回空闲队列队尾。注意没有清 hash：内容还在，还可能被 prefix cache 命中
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        # 返回：前缀命中了几个 block；如果空闲 block 不够分配，返回 -1
        #
        # 伪代码：
        #   h = -1
        #   for 第 i 个 block（最后一个除外）:
        #       h = hash(h, 这个 block 的 token)
        #       if hash 表里没有 h，或者 token 对不上: 停
        #       命中数 += 1
        #       if 命中的 block 正在被别的序列用: 不需要从空闲队列拿新 block
        #   空闲 block 够不够 "总 block 数 - 正在被用的命中 block 数"？
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # 最后一个 block 永远不查：它可能没写满；而且至少要留一个 token 去算，才能得到下一个 token 的 logits
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                # 命中的 block 正在被别人用：直接共享（ref_count + 1），不占空闲 block
                # （如果命中的 block 已经在空闲队列里，捞回来也要占用一个空闲名额，所以不减）
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 给序列建 block_table：前 num_cached_blocks 个复用缓存，后面的分配新 block
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1                 # 和别的序列共享
            else:
                block.ref_count = 1                  # 从空闲队列里"捞回来"
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 命中的前缀已经有 KV 了，prefill 从这里开始算
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        # 序列结束或被抢占：每个 block 引用计数 - 1，降到 0 的回到空闲队列
        # 倒序释放：后面的 block 先进空闲队列 → 先被覆盖，前面的（更可能被别人共享的前缀）留得更久
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # decode 前检查：len(seq) 已经包含了上一步采样出的 token（它的 KV 要在这一步写入）。
        # 如果 len(seq) % 256 == 1，说明这个 token 是一个新 block 的第一个 token → 需要 1 个空闲 block
        # （True/False 在比较时当作 1/0）
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # 每一步之后调用：这一步刚刚写满的 block，算 hash 并登记进 prefix cache
        # start/end：这一步之前和之后，"已写满的 block 数"
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1   # 链式 hash 的上一环
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
