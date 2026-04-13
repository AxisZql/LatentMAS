# 第 11 课：LatentMAS 核心原理

> 本节课时长：约 60-90 分钟
> 学习目标：深入理解 LatentMAS 的核心创新，掌握潜在空间推理的原理

---

## 11.1 回顾与引入

上节课我们学习了 TextMAS，它通过**文本传递**实现多 Agent 协作。本节课我们将学习 LatentMAS 的**核心创新**：在**潜在空间**传递信息，大幅减少 Token 消耗。

### 方法演进图

```
Baseline（单 Agent）
    │
    └──► TextMAS（文本传递）
              │
              │  问题：Token 消耗大
              │
              └──► LatentMAS（潜在空间传递）
                        │
                        │  核心创新：KV Cache 传递隐向量
                        │  优势：Token 消耗减少 50-80%
                        │
                        ▼
                   效率 + 效果双赢
```

### 本课学习路径

```
为什么需要潜在空间？
    │
    ├─── Token 消耗问题
    │
    ├─── 潜在空间是什么？
    │
    ├─── KV Cache 通信机制
    │
    └─── 潜在空间重对齐
```

---

## 11.2 为什么需要潜在空间推理？

### 11.2.1 TextMAS 的 Token 消耗问题

```
TextMAS 执行流程：

问题 → Planner → Critic → Refiner → Judger → 答案
        ↓         ↓         ↓         ↓
      50 tok   50 tok   50 tok   100 tok
      (文本)   (文本)   (文本)   (答案)
      ─────────────────────────────────────
              总计: ~250 tokens
```

**问题**：
- 每个 Agent 都需要生成完整文本
- 文本在 Agent 间传递，累积消耗大量 Token
- 推理速度慢，成本高

### 11.2.2 LatentMAS 的解决思路

**核心洞察**：Agent 之间传递的信息是**语义**而非**文本**。

```
传统：传递文本（high-level）
"第一步：理解问题..."

潜在空间：传递语义（low-level）
[h_1, h_2, h_3, ..., h_n]  ← 隐藏状态向量
```

**优势**：
| 对比项 | TextMAS | LatentMAS |
|--------|---------|-----------|
| Token 消耗 | ~250 | ~50-100 |
| 信息载体 | 文本 | 隐向量 |
| 传递内容 | 显式文本 | KV Cache |

---

## 11.3 潜在空间是什么？

### 11.3.1 Transformer 的隐藏状态

在 Transformer 中，每个 token 都有对应的**隐藏状态（Hidden State）**：

```
输入: [Token A] [Token B] [Token C]
          ↓         ↓         ↓
隐藏:   [h_A]    [h_B]    [h_C]     ← 潜在向量
         4096-d   4096-d   4096-d
```

**关键发现**：
- 最后一个 token 的隐藏状态 `h_C` 包含了整个序列的语义信息
- 这个向量就是"潜在表示"（Latent Representation）

### 11.3.2 潜在空间的优势

| 优势 | 说明 |
|------|------|
| **压缩** | 一个向量（4096 维）vs 一段文本（数百 Token）|
| **语义丰富** | 包含上下文信息，不只是字面意思 |
| **高效** | 只需传递向量，无需传递文本 |

### 11.3.3 工作记忆类比

```
人类工作记忆：
- 文本传递 = 说出来 + 听进去（慢）
- 潜在传递 = 心灵感应（快）

LLM 潜在空间：
- TextMAS = 文本传递（Token 消耗大）
- LatentMAS = 潜在传递（Token 消耗小）
```

---

## 11.4 KV Cache 通信机制

### 11.4.1 KV Cache 是什么？

KV Cache 是 Transformer 生成时缓存的 Key-Value 注意力矩阵：

```
Transformer 层注意力计算：

Query (Q) × Key (K)^T → Attention weights
                     ↓
              Value (V) 加权求和

KV Cache = 缓存的 K 和 V 矩阵
```

### 11.4.2 LatentMAS 的创新

**传统 KV Cache 用途**：加速自回归生成

**LatentMAS 用途**：在 Agent 间传递信息

```
LatentMAS Agent 间通信：

Planner 输出:
    → hidden_state → 存入 KV Cache
                        ↓
                   Critic 读取:
                        ← KV Cache → 生成反馈
                            ↓
                        Refiner 读取:
                            ← KV Cache → 生成改进
                                ↓
                            Judger 读取:
                                ← KV Cache → 生成答案
```

### 11.4.3 代码层面的实现

```python
# latent_mas.py
for agent in self.agents:
    if agent.role != "judger":
        # 非 Judger Agent：潜在空间生成
        past_kv = self.model.generate_latent_batch(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=self.latent_steps,  # 潜在推理步数
            past_key_values=past_kv,         # 传入历史 KV Cache
        )
    else:
        # Judger Agent：最终文本生成
        generated_batch, _ = self.model.generate_text_batch(
            judger_ids,
            judger_mask,
            max_new_tokens=self.judger_max_new_tokens,
            past_key_values=past_kv,  # 传入累积的 KV Cache
        )
```

---

## 11.5 潜在空间重对齐（Latent Realignment）

### 11.5.1 为什么需要重对齐？

Transformer 的结构：

```
Input Embedding Layer     Transformer Layers      Output Head
      ↓                        ↓                      ↓
[W_in]                    [Layers]               [W_out]
      ↓                        ↓                      ↓
  token → hidden_state → ... → hidden_state → logits
```

**问题**：
- 输入嵌入 `W_in` 和输出嵌入 `W_out` 是**不同的权重**
- 在潜在空间生成的向量来自 `W_out`，但需要投影回 `W_in` 的空间
- 不对齐会导致生成乱码

### 11.5.2 重对齐矩阵

```python
def _build_latent_realign_matrix(self, model, device, args):
    # 获取输入和输出嵌入权重
    input_weight = model.get_input_embeddings().weight   # [vocab, hidden]
    output_weight = model.get_output_embeddings().weight  # [hidden, vocab]

    # Gram 矩阵 + 正则化
    gram = torch.matmul(output_weight.T, output_weight)  # [hidden, hidden]
    gram = gram + 1e-5 * torch.eye(gram.shape[0])

    # 求解: M = (W_out^T @ W_out + λI)^(-1) @ W_out^T @ W_in
    rhs = torch.matmul(output_weight.T, input_weight)
    realign_matrix = torch.linalg.solve(gram, rhs)

    return realign_matrix, target_norm
```

### 11.5.3 应用重对齐

```python
def _apply_latent_realignment(self, hidden: torch.Tensor, model):
    # 投影到输入空间
    aligned = torch.matmul(hidden.float(), self.realign_matrix)

    # 范数对齐
    aligned_norm = aligned.norm(dim=-1, keepdim=True)
    aligned = aligned * (self.target_norm / aligned_norm)

    return aligned.half()
```

---

## 11.6 latent_steps 参数

### 11.6.1 作用

`latent_steps` 控制潜在空间推理的步数：

```python
def generate_latent_batch(self, input_ids, latent_steps=10, ...):
    # latent_steps = 10 表示在潜在空间推理 10 步
```

### 11.6.2 推理过程

```python
last_hidden = model(input_ids).hidden_states[-1]  # 初始隐藏状态

for step in range(latent_steps):
    # 1. 重对齐
    latent_vec = apply_realignment(last_hidden)

    # 2. 用潜在向量生成下一个表示
    outputs = model(inputs_embeds=latent_vec, past_key_values=past_kv)

    # 3. 更新 KV Cache 和隐藏状态
    past_kv = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1]
```

### 11.6.3 参数调优

| latent_steps | 效果 | 适用场景 |
|-------------|------|----------|
| 0 | 等同于 baseline | 快速实验 |
| 5-10 | 适中 | 一般任务 |
| 20-30 | 深度推理 | 复杂推理 |
| > 30 | 可能过拟合 | 谨慎使用 |

---

## 11.7 完整执行流程

### 11.7.1 Agent 循环

```python
for agent in self.agents:
    if self.args.prompt == "sequential":
        batch_messages = [
            build_agent_message_sequential_latent_mas(...)
        ]
    else:
        batch_messages = [
            build_agent_message_hierarchical_latent_mas(...)
        ]

    prompts, input_ids, attention_mask, tokens_batch = model.prepare_chat_batch(...)

    if agent.role != "judger":
        # Planner / Critic / Refiner：潜在空间推理
        past_kv = model.generate_latent_batch(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=self.latent_steps,
            past_key_values=past_kv,
        )
    else:
        # Judger：最终文本生成
        generated_batch, _ = model.generate_text_batch(
            judger_ids,
            judger_mask,
            past_key_values=past_kv,  # 传入累积的 KV Cache
        )
```

### 11.7.2 流程图

```
LatentMAS 执行流程：

问题 → [Planner] → [Critic] → [Refiner] → [Judger] → 答案
         ↓           ↓           ↓           ↓
      潜在推理    潜在推理    潜在推理    文本生成
         ↓           ↓           ↓           ↓
      KV Cache   KV Cache   KV Cache   输出答案
         └───────────┴───────────┘
                    ↓
               累积传递
```

### 11.7.3 Token 消耗对比

| 方法 | Planner | Critic | Refiner | Judger | 总计 |
|------|---------|--------|---------|--------|------|
| TextMAS | 50 | 50 | 50 | 100 | ~250 |
| LatentMAS | 10 | 10 | 10 | 100 | ~130 |
| 节省 | 80% | 80% | 80% | - | ~50% |

---

## 11.8 与 TextMAS 的关键区别

### 11.8.1 信息传递方式

| 对比项 | TextMAS | LatentMAS |
|--------|---------|-----------|
| 传递内容 | 文本字符串 | KV Cache 隐向量 |
| 存储位置 | 上下文变量 | 模型内部状态 |
| 处理方式 | 拼接到 prompt | 传递给 `past_key_values` |
| Token 消耗 | 高 | 低 |

### 11.8.2 提示词差异

```python
# TextMAS Critic 提示词
"""
You are provided with:
(1) the original question
(2) the Planner Agent's plan in text format.
"""

# LatentMAS Critic 提示词
"""
The plan information is provided in latent KV representation format.
"""
```

### 11.8.3 架构选择

| 架构 | 特点 | 适用场景 |
|------|------|----------|
| **Sequential** | 串行协作，强调反馈 | 需要深度反思的任务 |
| **Hierarchical** | 分层专家，独立决策 | 多领域综合任务 |

---

## 11.9 核心创新总结

### 11.9.1 三大创新点

| 创新点 | 说明 | 效果 |
|--------|------|------|
| **潜在空间通信** | Agent 间传递隐向量而非文本 | Token 消耗减少 50-80% |
| **KV Cache 复用** | 累积 past_key_values | 推理加速 3-7x |
| **无训练对齐** | Gram 矩阵求逆实现空间对齐 | 稳定生成，无需微调 |

### 11.9.2 性能提升

```
LatentMAS 性能：

准确率：与 TextMAS 相当或更好
Token：减少 50-80%
速度：提升 3-7x
显存：更低（KV Cache 复用）
```

---

## 11.10 本课小结

### 核心要点

| 要点 | 内容 |
|------|------|
| **潜在空间** | Transformer 隐藏状态，压缩语义信息 |
| **KV Cache 通信** | 通过 past_key_values 传递隐向量 |
| **潜在空间重对齐** | Gram 矩阵求逆，投影回输入空间 |
| **latent_steps** | 控制潜在推理深度 |
| **核心优势** | Token 消耗少，速度快 |

### LatentMAS vs TextMAS

| 对比项 | TextMAS | LatentMAS |
|--------|---------|-----------|
| 通信方式 | 文本传递 | 潜在向量传递 |
| Token 消耗 | ~250 | ~130 |
| 推理速度 | 较慢 | 快 3-7x |
| 显存占用 | 较高 | 较低 |
| 实现复杂度 | 较低 | 较高 |

---

## 11.11 课后练习

1. **理解原理**：用自己的话解释为什么潜在空间通信能减少 Token
2. **参数调优**：尝试不同的 `latent_steps` 值，观察效果变化
3. **对比实验**：对比 TextMAS 和 LatentMAS 的 Token 消耗
4. **思考问题**：
   - 为什么需要潜在空间重对齐？
   - LatentMAS 的局限性是什么？

---

## 下节课预告

> **第 12 课：LatentMAS 实现详解**
> - `generate_latent_batch()` 详解
> - `_truncate_past()` KV Cache 裁剪
> - `run_batch()` vs `run_batch_vllm()` 的区别
> - 代码走读：完整执行流程

---

*本教程由 Claude Code 生成，保存于 `./tutorial/` 目录*
