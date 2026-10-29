# =============================================================================
# loader.py —— 从 HF 的 .safetensors 文件加载权重
#
# 伪代码：
#   for 每个 .safetensors 文件, for 文件里的每个权重名:
#       if 名字里有 q_proj / k_proj / v_proj / gate_proj / up_proj（packed_modules_mapping 里的键）:
#           换成合并后的参数名（比如 q_proj → qkv_proj），调用它的 weight_loader，并告诉它是哪一段
#       else:
#           找到同名参数，调用它的 weight_loader（没有就直接拷贝）
#
# weight_loader 是各个层挂在参数上的函数（见 layers/linear.py），负责 TP 切分和"拷进合并矩阵的哪一段"。
# =============================================================================
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:      # 先读到 CPU，copy_ 时再拷到 GPU 上的参数里
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # for ... else：内层 for 没有 break（不是合并参数）时执行
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
