# 学习路线：一步一步走到 showcase 分支

`showcase` 分支是在 upstream `main` 之上的一串 commit，**每个 commit 都能单独 checkout、跑通**。
按顺序学：每一步先读"做了什么 / 为什么"，再读代码，再跑一遍，最后能回答"自测问题"，就算吃透了这一步。

```bash
cd ~/nano-vllm && git log --oneline --reverse main..showcase    # 整条路线
git show <commit> --stat                                         # 这一步改了哪些文件
git worktree add ~/learn <commit>                                # 在单独目录里停在某一步，随便改、随便跑
```

跑脚本时记得让 Python 用当前目录的代码：`PYTHONPATH=$PWD python perf/xxx.py`（venv 里 `pip install -e` 指向的是 `~/nano-vllm`）。
测试不用管这个，`pyproject.toml` 里已经配好了。

---

## 第 1 步　`perf: baseline on RTX 5070 Ti (sm_120) vs SGLang`

**做了什么**：在 5070 Ti 上跑通，测 `bench.py` 的吞吐（4543 tok/s），和 SGLang 0.5.22 对比。

**读**：`perf/BASELINE.md`，`perf/bench_sglang.py`。

**自测**
1. SGLang 默认配置为什么慢 14%？（提示：它在这张卡上只为 bs ≤ 8 捕获 CUDA graph，而且给 KV cache 留的显存更少）
2. 这个 benchmark 测的是吞吐还是延迟？它说明不了什么？

## 第 2 步　`Add profile_bench.py`

**做了什么**：不改引擎代码，用"包一层方法"（monkeypatch）的办法给引擎插桩：每一步的 CPU 时间、GPU 时间（CUDA event）、batch 大小、抢占次数、重算了多少 token。

**读**：`perf/profile_bench.py`（重点：`timed_cpu` / `timed_gpu` 两个包装器，`preempt` 怎么统计重算），然后读结果 `perf/results/profile_full.txt`。

**结论（面试要能脱口而出）**
- 86% 的时间在 decode，GPU 忙碌 96.5%，所以**不是 CPU 调度的问题**。
- KV cache 只有 415 块，256 条请求同时跑需要 1206 块 → **138 次抢占**，30% 的 prefill 是在重算被抢占的序列，decode 的平均 batch 只有 69。

**自测**
1. 为什么 GPU 时间要用 CUDA event，不能直接用 `time.perf_counter()`？（GPU 异步执行；CPU 计时只量到"发射"）
2. 引擎每一步在哪里同步 CPU 和 GPU？（sampler 输出 `.tolist()`）
3. 文件末尾那个线性拟合给出的带宽是 1069 GB/s，超过了显存峰值 896 GB/s，为什么不可信？（batch 大小和上下文长度高度共线，回归分不开两者）

## 第 3 步　`Add decode microbenchmarks`

**做了什么**：单独测 attention kernel 和整个 decode step。`bench_decode.py` 用 flash-attn 测 attention 占一步的比例和实际带宽；`bench_flashinfer_fp8.py` 比较 FlashInfer 的 bf16 / fp8 decode kernel。

**读**：两个脚本和 `perf/results/bench_decode.txt`、`bench_flashinfer_fp8.txt`。

**结论**：在 bench.py 的工作点（bs≈90、ctx≈900）上，attention 占 decode 的 70–80%，flash-attn 已经跑到峰值带宽的 85–93%。**kernel 再快也快不了多少，只能少读字节** → fp8。

**自测**
1. 手算：bs=128、ctx=1024 时，一层 attention 要读多少字节的 KV？（128 × 1024 × 2(K,V) × 8 heads × 128 dim × 2 bytes = 512 MiB……再除以时间就是带宽）
2. 表里带 `*` 的行带宽超过了峰值，原因是什么？（多个序列共享 block，并发读时命中 L2）
3. 为什么权重不是瓶颈？（1.2 GB 权重每步读一次，被整个 batch 分摊；KV 每个 token 112 KiB，不能分摊）

## 第 4 步　`Add FP8 (e4m3) KV cache`（核心）

**做了什么**：`kv_cache_dtype="fp8"` 时 KV cache 用 `float8_e4m3fn` 存，容量翻倍；decode attention 换成能直接读 fp8 的 FlashInfer。

**读的顺序**
1. `nanovllm/config.py`：两个新参数。
2. `nanovllm/layers/attention.py`：`store_kvcache_kernel` 里的 `x / scale`、`clamp(±448)`，以及 `forward` 里的分支。
3. `nanovllm/engine/model_runner.py`：`allocate_kv_cache`（为什么按 `kv_dtype.itemsize` 算块数）→ `prepare_decode` → `run_model`（plan 之后再 replay）→ `capture_cudagraph`（每个 batch size 一个 wrapper）。

**关键概念**
- e4m3：1 位符号、4 位指数、3 位尾数，最大 448，**没有 inf**。
- FlashInfer 的"先 plan 后 run"：plan 在 CPU 上为这一批请求分工，写进 workspace；run 每层调一次。
- CUDA graph 把 kernel 读的**地址**固定下来，所以每个 graph 要有自己的固定 buffer，每次 replay 前重新 plan，往这些 buffer 里写新内容。

**自测**（第 4 步最重要，每题都要能讲清楚）
1. 为什么每个 graph batch size 一个 wrapper，不能共用一个？
2. 真实 batch 是 100、graph 是 112 时，多出来的 12 个假请求读什么、写什么？为什么不会污染真实数据？（读 block 0 的 1 个 token；写 slot −1，store kernel 直接跳过；输出被切掉）
3. Qwen3-0.6B 第 0 层的 key 最大到 ~530，直接 cast 会怎样？为什么默认 k_scale 是 2？
4. 为什么 flash-attn 不行，非要换 FlashInfer？

## 第 5 步　`Add eval_kv_quality.py and FP8 KV cache results`

**做了什么**：衡量 fp8 掉了多少质量。16 段文本，先分两块 prefill（第二块走 prefix cache 路径），再**teacher forcing**：每一步喂真实的下一个 token，记录真实 token 的 NLL 和 argmax。

**自测**
1. 为什么用 teacher forcing，不直接比较两边生成的文本？（生成一旦有一个 token 不同，后面全不同，误差会累积；teacher forcing 让两边输入完全相同）
2. NLL、ppl、argmax agreement 各说明什么？ppl = exp(平均 NLL)。

## 第 6 步　`Add nsys_profile.py`

**做了什么**：用 NVTX 给引擎的各个阶段打标签，配合 Nsight Systems 看时间线。

**练习**（在 Mac 上用 Nsight Systems 打开 `perf/results/nsys_bf16.nsys-rep` 和 `nsys_fp8.nsys-rep`）
1. 找一个 decode step：GPU 上持续多久？相邻两步之间 GPU 空闲多久？
2. 一层 attention kernel 有多宽？bf16 和 fp8 各量一次，算比值。
3. `run_model` 之后 CPU 在等什么？（`.tolist()` 触发的 `cudaMemcpyAsync`：异步 GPU 工作的时间被算到了同步点上）
4. 为什么 fp8 的步间空隙看起来更明显？（GPU 部分变快了，CPU 的固定开销占比变大；每步还多了一次 `plan()`）

## 第 7 步　`fp8: FlashInfer paged prefill for prefix-cache hits; share workspaces; tests`

**做了什么**
- 第 4 步里 prefix cache 命中时，是先把要读的 block **反量化成一份临时 bf16 拷贝**再给 flash-attn。这份拷贝**没有计入 KV cache 的显存预算**（最坏约 1 GB，可能 OOM）。现在改用 FlashInfer 的 `BatchPrefillWithPagedKVCacheWrapper` 直接读 fp8，质量不变（NLL 1.9843 → 1.9842）。
- 所有 wrapper（eager、prefill、37 个 graph）共享同一块 workspace。FlashInfer 原本给每个 wrapper 单独分配 8 MB，37 个 graph 一共约 300 MB。
- 把 `paged_kv_layout()` / `plan_attention()` 提成独立函数，**让测试能调用引擎自己的代码**。
- 加了单元测试。

**读**：`model_runner.py` 开头两个函数，`tests/test_fp8_attention.py`。

**自测**
1. 手算 `paged_kv_layout`：block_size=256，两条序列长 300 和 256，block 表是 [7, 3] 和 [5]。indptr、indices、last_page_len 各是什么？（[0, 2, 3]，[7, 3, 5]，[44, 256]）
2. 为什么多个 wrapper 共享 workspace 是安全的？什么情况下会不安全？（同一时刻只有一个在跑，跑之前一定重新 plan；如果两个 graph 并发跑，或者跳过 plan 直接 replay，就不安全）
3. 测试里的 float32 reference 是怎么算的？为什么要用 graph"轮流 replay"来测？

## 第 8 步　`fp8: per-layer KV scales from a calibration run`

**做了什么**：`kv_cache_scales` 可以是一个 JSON，每层一对 scale。`perf/calibrate_kv_scales.py` 在 40 段**和评测不重叠**的文本上统计每层存进 cache 的最大 |k|、|v|，除以 448 再乘 1.25 的余量。

**最值得讲的发现**

| scales | NLL 相对 bf16 | argmax 一致率 |
|---|---|---|
| (1, 1) 全局 | +0.0123 | 95.08% |
| (2, 1) 全局（默认） | +0.0135 | 95.23% |
| 每层校准 | **−0.0007** | 95.36% |

- **偏差（bias）**被修掉了：前几层的 V 很小（第 1 层 |v| ≤ 1.0）。全局 v_scale=1 时，大部分值落在 e4m3 的**次正规数**区间（< 2⁻⁶ = 0.0156），相对精度急剧下降。按层缩放把每层的数值范围映射到正规数区间。
- **噪声（noise）**没变：3 位尾数意味着每个值都有最多约 6% 的舍入误差。平均下来 NLL 不变，但在 top-2 几乎打平的位置会翻转 argmax（约 5%）。要消除只能加位数，换 scale 没用。

**自测**
1. 浮点格式里"scale 只影响范围、不影响相对精度"这句话什么时候成立，什么时候不成立？（正规数区间内成立；落进次正规区或超过最大值时不成立）
2. 为什么校准文本必须和评测文本不重叠？
3. 为什么默认值还保留全局 (2, 1)，不直接用校准结果？（全局值不需要校准、对任何模型都能用；校准结果和模型绑定）

## 第 9 步　`tests: end-to-end engine tests`

**做了什么**：每种配置起一个独立进程跑引擎。测的内容：fp8 的 block 数 ≥ bf16 的 1.9 倍；chunked prefill 和 prefix 命中确实走了 FlashInfer；事实类问题答对；teacher forcing 质量在界内；CUDA graph 和 eager 一致。

**两个调试中学到的东西**（面试很加分，说明你真的踩过坑）
1. **"greedy" 不是确定的**：sampler 不允许 temperature=0，用 1e-5 的话，bf16 的 logits 会出现完全相等的平局，sampler 随机打破平局。所以**同一个配置跑两次**生成结果都可能不一样，不能逐 token 比较生成结果。
2. **CUDA graph 和 eager 什么时候 bit-identical**：batch 刚好填满一个 graph（16 条）时完全一致；有 padding 时（5 条请求用 bs=8 的 graph）不一致。原因是 FlashInfer 根据 batch size 决定怎么切分 KV 的工作，切分不同，浮点加法的顺序就不同。bf16 的 flash-attn 也一样。测试断言的是"这个差异远小于 fp8 本身的影响"。

## 第 10 步　`perf: sweep` + 文档

**做了什么**：并发请求数从 8 到 512 扫一遍，bf16 和 fp8 各跑两遍取平均：`perf/sweep.py`，结果在 `perf/results/sweep.png`。完整的英文文档在 `docs/fp8_kv_cache.md`。

**怎么读这张图**（数字见 `perf/results/sweep.txt`）

- **n ≤ 128**：两边都没有抢占，decode batch 一样大，提升（1.08× → 1.47×）全部来自带宽：attention 少读了一半字节。n 越大提升越多，因为 attention 在一步里的占比随 batch × 上下文增长；n=8 时一步的大头是读权重和启动 kernel。
- **n = 256**：bf16 的 KV 不够了（132 次抢占，吞吐反而比 n=128 低），fp8 还装得下（9 次），decode batch 72 → 138，两种收益叠加：1.85×。
- **n = 512**：fp8 也不够了（279 次抢占）→ 1.60×。容量翻倍只是把"悬崖"往后推；再往后，瓶颈变成调度器的抢占策略（丢掉 KV，之后重算）。

**自测**：n=8 时只快 8%，为什么？如果面试官问"小 batch、在线低延迟场景下 fp8 值不值"，你怎么回答？

---

## 面试怎么讲

**30 秒版本**

> 我在 nano-vllm（一个约 1200 行的 vLLM 简化实现）上做了 FP8 KV cache。先 profile：decode 占 86% 的时间，其中 attention 占 70–80%，而且 flash-attn 已经跑到显存带宽的 90%，KV cache 不够还导致了 138 次抢占。所以瓶颈是每个 token 的 KV 字节数。我把 KV cache 改成 e4m3，attention 换成能直接读 fp8 的 FlashInfer，处理了 CUDA graph 和 padding。在 5070 Ti 上吞吐提升 1.78 倍；按层校准 scale 之后，perplexity 和 bf16 持平。

**常见追问（每题准备两三句话）**
1. 为什么选 FP8，而不是 INT8 或者 INT4？（e4m3 是浮点，相对误差在整个正规数区间里都差不多，对离群值不敏感，一个 per-tensor scale 就够；INT8 是均匀量化，离群值会把小值的精度挤掉，通常要更细粒度的 scale；FlashInfer 原生支持 fp8 KV；INT4 要分组 scale，kernel 也更复杂）
2. 量化误差来自哪里？怎么测的？（bias 和 noise 的区分；teacher forcing）
3. CUDA graph 怎么和 FlashInfer 配合？（固定地址的 buffer、每次 replay 前重新 plan、padding）
4. 怎么证明结果是对的？（单测对 float32 reference、bit-exact 的 store、端到端测试、bf16 路径和 upstream 输出完全一致）
5. 小 batch 时有提升吗？为什么？（看 sweep：显存不是瓶颈时只剩带宽收益，而且 plan 的 CPU 开销占比更大）
6. 还能怎么做？（per-head 或 per-block 动态 scale；把 plan 和上一步的 GPU 计算重叠；prefill 和 decode 混合调度；TP 下验证）
7. 测量时怎么保证可信？（交替跑、记录 GPU 状态、关掉其他占用 GPU 的程序；带宽超过峰值时知道是 L2 命中造成的假象）

**简历 bullet（中英各一）**

- 为 nano-vllm 实现 FP8（e4m3）KV cache：基于 profiling 定位 decode 阶段受 KV 读取带宽和容量限制，用 Triton 写量化 kernel，接入 FlashInfer 的 paged attention 并适配 CUDA graph；RTX 5070 Ti 上吞吐提升 1.78×，抢占从 138 次降到 0，按层校准 scale 后 perplexity 与 bf16 持平（ΔNLL −0.0007）。
- Implemented an FP8 (e4m3) KV cache in nano-vllm: profiled decode as KV-bandwidth- and capacity-bound, wrote the quantizing store kernel in Triton, and integrated FlashInfer paged attention with CUDA graphs. 1.78× throughput on an RTX 5070 Ti, preemptions 138 → 0, perplexity on par with bf16 after per-layer scale calibration.
