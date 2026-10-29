# =============================================================================
# embed_head.py —— 词嵌入（token id → 向量）和 LM head（向量 → 每个词的 logits）
#
# TP 时按词表切：每张卡只存词表的 1/tp_size。
#   Embedding：不属于本卡范围的 token 先查到 0 号位置再乘 mask 清零，最后 all_reduce 把各卡结果加起来
#   LM head：每张卡算自己那段词表的 logits，再 gather 到 rank 0 拼起来
#
# LM head 的一个重要优化：prefill 时只需要每个序列"最后一个 token"的 logits
# （只有它用来采样下一个 token），所以先挑出这些位置再做矩阵乘。
# 词表有 15 万个词，对一个 1000 token 的 prompt 只算 1 行而不是 1000 行，省很多计算。
# =============================================================================
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank   # 本卡负责的词表范围
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)    # 不归本卡的 token 变成 0（随便查一个，后面会清零）
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)                       # 每个 token 只有一张卡给出非零结果，加起来就是正确的
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # cu_seqlens_q = [0, 3, 8] → 每个序列最后一个 token 的下标是 [2, 7]
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)            # [num_seqs, vocab / tp_size]
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
