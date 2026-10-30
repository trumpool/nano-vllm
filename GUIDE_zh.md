# nano-vllm 入门讲解（先大白话，再对代码）

读法建议：

1. 先读**第一部分**，直到你能用自己的话讲出每个部件解决什么问题。
2. 运行 `tools/trace_engine.py`，对照**第二部分**看每一步发生了什么。
3. 最后才打开代码，只看**第三部分**表里对应的那个函数，配合代码里的中文注释读。
4. 改参数做实验：**先写下预测，再运行**。预测错的地方，就是还没真正理解的地方。

---

## 第一部分：大白话

### 0. 大模型怎么"写字"

模型只会做一件事：**看一串字，预测下一个字**。生成一段话就是不断重复：

```
"今天天气"     → "很"
"今天天气很"   → "好"
"今天天气很好" → "。"
```

推理引擎要解决的是：**怎么又快又省地同时服务很多人**。下面每一节是"一个问题 → 一个部件"。

### 1. 每生成一个字都把前文重读一遍，太浪费 → KV cache

模型读每个字时会算出一份"笔记"（K 和 V）。后面的字回头参考前文时，**查笔记就行，不用重读**。

- **prefill（预填充）**：第一次读完整个 prompt，给每个字记笔记。一次处理几百上千个字，**计算量大**。
- **decode（解码）**：之后每步只处理 1 个新字：查所有旧笔记，再补一条新笔记。**计算很少，但要把所有笔记从显存读一遍**——典型的 memory-bound。

### 2. 一次只服务一个人，GPU 大部分时间闲着 → 批处理 + 调度器（Scheduler）

把很多人的请求凑成一批一起算：256 个人各算 1 个字，和 1 个人算 1 个字花的时间差不多。

**continuous batching**：批不是固定的。有人生成完就马上离开，新请求马上加入。

调度器每一步决定"这一步算谁"。nano-vllm 的规则：
- **有新请求要读 prompt，就优先 prefill**（这一步只做 prefill）
- 没有时，才让所有生成中的请求各前进 1 个字（decode）

### 3. 每人的笔记越写越多，事先不知道多长 → 分页（BlockManager）

和操作系统管内存一样：**笔记本切成固定大小的"页"（block），每页记 256 个字**。按需发页，页不需要连续。每个请求有一张**页表**（`block_table`），比如 `[0, 1, 3]` 表示它的第 0/1/2 页是笔记本的第 0/1/3 页。

### 4. 页发完了还有人要写 → 抢占（preemption）

某个请求写满当前页、需要新页但没有空页时：把**最后加入的请求请出去**，收回它所有的页，让它回等待队列，之后**重新读一遍**（重新 prefill）。

### 5. 很多请求开头一样（同一个 system prompt），每人记一遍很浪费 → 前缀缓存（prefix caching）

每页写满后算一个**指纹**（hash），由"前一页的指纹 + 这一页的内容"算出，所以**指纹相同 ⇔ 从开头到这一页的内容全部相同**。新请求按指纹一页页查，查到就**共享那一页**（引用计数 +1），不用重算。

只有**写满的页**才有指纹，所以复用的粒度是整页：共享 600 个字的前缀，只能复用前 2 页（512 个字）。

### 6. 一个超长 prompt 把一整步占满 → chunked prefill（分块预填充）

每步有 token 预算（默认 16384）。超长的 prompt 切成几段、分几步读完。nano-vllm 里只有一批中的第一个请求能被切。

### 7. decode 一步要启动几百个小 kernel，CPU 发号施令太慢 → CUDA graph

把一次 decode 的所有 kernel **录成一段录像**，之后每步**直接重放**，省掉每个 kernel 的启动开销。prefill 不用（每步形状都不同，而且计算量大，启动开销占比小）。

### 8. 长短不一的请求怎么一起送进模型

- **varlen**：把这一步所有请求的字**首尾拼成一长串**（不补 padding），用 `cu_seqlens` 记录每个请求的起止位置
- **slot_mapping**："这一步每个新字的笔记，要**写到**笔记本的哪个格子"（格子编号 = 页号 × 256 + 页内位置）
- **block_table**："每个请求的旧笔记，要去笔记本的哪几页**读**"

---

## 第二部分：亲眼看它运行

```bash
cd ~/nano-vllm && source ~/venvs/nanovllm/bin/activate
python tools/trace_engine.py
```

默认场景：KV cache 只有 6 页，每步最多 512 个字；A（766 字）、B（612 字，前 512 字和 A 相同）、C（400 字）。输出里能看到：

| step | 发生了什么 | 对应第一部分 |
|---|---|---|
| 1 | A 只读了前 512 个字，还留在 waiting | 6. chunked prefill |
| 2 | B 命中 A 的前 2 页，页表 `[0, 1, 3]` 和 A 的 `[0, 1, 2]` 前两项相同，只算 100 个字 | 5. 前缀缓存 |
| 3 | C 读完 prompt，6 页全部用完 | 3. 分页 |
| 4–5 | 三个人一起 decode，每人每步 1 个字 | 2. 批处理 |
| 6 | A 写到第 768 个字需要第 4 页，C 被抢占，A 拿到 C 的一页 | 4. 抢占 |
| 8–9 | A 完成离开，C 马上回来重新 prefill；它之前的第 1 页内容还在、又命中了缓存，只需重算 147 个字 | 2. continuous batching + 5. 前缀缓存 |

改参数做实验（先预测再运行）：

```bash
python tools/trace_engine.py --blocks 5            # 只有 5 页：谁被抢占？什么时候？
python tools/trace_engine.py --prefix 600          # 共同前缀 600：能复用几页？
python tools/trace_engine.py --budget 2048         # 预算 2048：还会分块吗？前缀缓存还会命中吗？
python tools/trace_engine.py --help
```

---

## 第三部分：对应到代码

| 概念 | 代码位置 |
|---|---|
| 主循环：调度 → 执行 → 后处理 | `nanovllm/engine/llm_engine.py` 的 `step()`（3 行） |
| 请求的状态 | `nanovllm/engine/sequence.py` |
| prefill 优先、chunked prefill | `nanovllm/engine/scheduler.py` 的 `schedule()` 前半段 |
| decode、抢占 | `schedule()` 后半段、`preempt()` |
| 发页、收页、页表 | `nanovllm/engine/block_manager.py` 的 `allocate()`、`deallocate()`、`may_append()` |
| 前缀缓存（指纹） | `block_manager.py` 的 `can_allocate()`、`hash_blocks()` |
| 拼接（varlen）、slot_mapping、block_table | `nanovllm/engine/model_runner.py` 的 `prepare_prefill()`、`prepare_decode()` |
| KV cache 有多大 | `model_runner.py` 的 `allocate_kv_cache()` |
| 写笔记、读笔记 | `nanovllm/layers/attention.py` |
| CUDA graph | `model_runner.py` 的 `capture_cudagraph()`、`run_model()` |
| 模型结构 | `nanovllm/models/qwen3.py` |
