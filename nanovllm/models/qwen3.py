# =============================================================================
# qwen3.py —— Qwen3 模型结构（标准的 decoder-only Transformer）
#
#   Qwen3ForCausalLM
#   └── Qwen3Model
#       ├── embed_tokens                       token id → hidden（1024 维）
#       ├── layers × 28：Qwen3DecoderLayer
#       │     h, r = RMSNorm(h + r)            ← 融合了"加残差"和"归一化"
#       │     h    = Attention(h)
#       │     h, r = RMSNorm(h + r)
#       │     h    = MLP(h)
#       └── norm                               最后再加一次残差、归一化
#   └── lm_head                                hidden → 词表大小的 logits（只在 compute_logits 里调用）
#
# Attention 块：qkv_proj（一次矩阵乘出 Q/K/V）→ 每个头单独做 RMSNorm（q_norm / k_norm，Qwen3 特有）
#               → RoPE → attention → o_proj
# MLP 块（SwiGLU）：gate_up_proj（一次矩阵乘出 gate 和 up）→ SiLU(gate) × up → down_proj
#
# 残差的传法：普通写法是 x = x + sublayer(norm(x))。这里把 "x + ..." 推迟到下一个 RMSNorm 里做，
# 所以每层输入输出是 (hidden_states, residual) 两个张量，第一层时 residual 还是 None。
# =============================================================================
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        # TP 时头按卡均分：每张卡负责 num_heads / tp_size 个 Q 头和 num_kv_heads / tp_size 个 KV 头
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5          # attention 的 1/sqrt(d)
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            # Qwen3 去掉了 QKV 的 bias，改成对每个头的 q、k 各做一次 RMSNorm（QK-Norm），训练更稳定
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [N, hidden]，N = 本步所有新 token 的总数（多个序列拼在一起）
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)          # [N, 16, 128]
        k = k.view(-1, self.num_kv_heads, self.head_dim)       # [N, 8, 128]
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)                                 # 里面会写 KV cache
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)     # [N, 2 × intermediate]
        x = self.act_fn(gate_up)           # SiLU(gate) × up → [N, intermediate]
        x = self.down_proj(x)              # [N, hidden]
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层：还没有残差，先把输入本身存成残差
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 其他层：把上一层 MLP 的输出加进残差，再归一化
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # 把 attention 的输出加进残差，再归一化，送进 MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # 注意：MLP 的输出还没加回残差，留给下一层的 input_layernorm（或最后的 norm）去加
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)     # 加上最后一层 MLP 的残差，再归一化
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # HF checkpoint 里 q/k/v、gate/up 是分开存的，这里的模型把它们合并成了一个矩阵。
    # 加载时（utils/loader.py）按这张表把 HF 的权重名映射到合并后的参数，并告诉它是哪一段：
    #   "model.layers.0.self_attn.q_proj.weight" → qkv_proj 的 "q" 段
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 小模型常见的做法：输入嵌入和输出 lm_head 共用同一个权重矩阵（Qwen3-0.6B 就是）
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 和 forward 分开：这样 decode 的 CUDA graph 只录 forward，lm_head 在 graph 外面跑
        return self.lm_head(hidden_states)
