# =============================================================================
# linear.py —— 支持 Tensor Parallel（TP）的线性层
#
# 只有一张卡时（tp_size = 1），这些类都等价于普通的 F.linear(x, W)，可以先跳过 TP 的细节。
#
# TP 的两种切法（权重 W 的形状是 [out, in]，y = x Wᵀ）：
#
#   ColumnParallel：按"输出维"切（tp_dim = 0）。每张卡算 y 的一部分列，输入 x 每张卡都是完整的。
#       用在 QKV 投影、gate_up 投影这类"输出接下来会被各卡分头处理"的地方。
#
#   RowParallel：按"输入维"切（tp_dim = 1）。每张卡拿到 x 的一部分、算出一个部分和，
#       最后 all_reduce 把各卡的部分和加起来。用在 o_proj、down_proj。
#
#   Megatron 的经典配对：Column → (各卡独立算 attention / 激活) → Row → all_reduce，
#   这样一个 attention 块或 MLP 块只需要一次通信。
#
# weight_loader：从 HF 的 checkpoint（完整权重）里切出本卡的那一份，拷进参数。
# =============================================================================
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 把加载函数挂在参数上，utils/loader.py 加载权重时会找到并调用它
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    # 每张卡都存完整的权重（不切）

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)   # 每张卡只存 out / tp_size 行

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 从完整权重里取出第 tp_rank 段（沿输出维）
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    # 把几个输入相同的线性层（比如 MLP 的 gate_proj 和 up_proj）拼成一个大矩阵，
    # 一次矩阵乘算出来，比分开算两次更快（一个大 GEMM 比两个小 GEMM 更能喂饱 GPU）。
    # 权重布局：[gate | up]（TP 时每张卡是 [gate 的第 rank 段 | up 的第 rank 段]）

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # loaded_shard_id：checkpoint 里的这个权重是第几个子矩阵（0 = gate，1 = up），决定拷到大矩阵的哪一段
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    # Q、K、V 三个投影合并成一个矩阵乘。权重布局：[Q | K | V]
    # GQA：K、V 的头数比 Q 少（Qwen3-0.6B：16 个 Q 头，8 个 KV 头），所以三段长度不一样

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # 根据是 q / k / v，算出它在 [Q | K | V] 里的偏移和长度
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)   # 每张卡只存 in / tp_size 列

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:       # bias 不切，每张卡一份完整的
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias 只在 rank 0 加：all_reduce 会把各卡结果相加，如果每张卡都加 bias 就会被加 tp_size 次
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
