# 第 10 课：TextMAS 文本多智能体

> 本节课时长：约 60-75 分钟
> 学习目标：理解 TextMASMethod 的实现，掌握多 Agent 协作流程和上下文累积机制

---

## 10.1 回顾与引入

上节课我们学习了基线方法（baseline），它使用单个 Agent 直接生成答案。本节课我们将学习**第一种多 Agent 协作方法：TextMAS**，Agent 之间通过**文本传递信息**。

### 方法演进

```
Baseline（单 Agent）
    │
    └──► TextMAS（文本多 Agent）
              │
              └──► LatentMAS（潜在空间多 Agent）← 核心创新
```

### 本课学习路径

```
TextMASMethod
    │
    ├─── __init__ 初始化
    │
    ├─── run_batch 批量处理
    │
    ├─── Agent 循环
    │
    ├─── 上下文累积
    │
    └─── 结果构建
```

---

## 10.2 TextMASMethod 类概述

### 10.2.1 类定义

```python
class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()  # 获取 4 个 Agent
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task
```

### 10.2.2 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `self.agents` | List[Agent] | 4 个 Agent 的列表 |
| `self.max_new_tokens_each` | int | 每个 Agent 的最大生成长度 |
| `self.contexts` | List[str] | 累积的上下文 |

### 10.2.3 default_agents() 定义

```python
# methods/__init__.py
def default_agents() -> List[Agent]:
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Critic", role="critic"),
        Agent(name="Refiner", role="refiner"),
        Agent(name="Judger", role="judger"),
    ]
```

---

## 10.3 run_batch() 详解

### 10.3.1 函数签名

```python
def run_batch(self, items: List[Dict]) -> List[Dict]:
```

### 10.3.2 初始化

```python
batch_size = len(items)
contexts = ["" for _ in range(batch_size)]           # 当前上下文
history_contexts = ["" for _ in range(batch_size)]   # 历史上下文（完整）
agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]  # Agent 轨迹
final_texts = ["" for _ in range(batch_size)]       # 最终文本
```

**变量说明**：
| 变量 | 用途 |
|------|------|
| `contexts` | 当前 Agent 的上下文（会被下一个 Agent 使用）|
| `history_contexts` | 完整的历史上下文（记录所有 Agent 输出）|
| `agent_traces` | 每个样本的 Agent 执行轨迹 |
| `final_texts` | Judger 的输出（最终答案）|

---

## 10.4 Agent 执行循环

### 10.4.1 核心循环

```python
for agent in self.agents:
    # 1. 根据架构选择消息构建函数
    if self.args.prompt == "hierarchical":
        batch_messages = [
            build_agent_messages_hierarchical_text_mas(
                role=agent.role,
                question=item["question"],
                context=contexts[idx],
                ...
            )
            for idx, item in enumerate(items)
        ]
    else:
        batch_messages = [
            build_agent_messages_sequential_text_mas(
                role=agent.role,
                question=item["question"],
                context=contexts[idx],
                ...
            )
            for idx, item in enumerate(items)
        ]

    # 2. 准备输入张量
    prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
        batch_messages, add_generation_prompt=True
    )

    # 3. 批量生成
    if self.model.use_vllm:
        generated_texts = self.model.vllm_generate_text_batch(...)
    else:
        generated_texts, _ = self.model.generate_text_batch(...)

    # 4. 处理每个样本的输出
    for idx in range(batch_size):
        text_out = generated_texts[idx].strip()

        # 格式化输出
        if self.args.prompt == "hierarchical":
            formatted_output = f"[{agent_name_map[agent.name]}]:\n{text_out}\n\n"
        else:
            formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

        # 更新上下文（除了 Judger）
        if agent.role != "judger":
            contexts[idx] = f"{contexts[idx]}{formatted_output}"
            history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
        else:
            final_texts[idx] = text_out  # Judger 输出作为最终答案

        # 记录 Agent 轨迹
        agent_traces[idx].append({...})
```

### 10.4.2 执行流程图

```
Agent 循环执行流程：

初始化：contexts = ["", "", ...]

┌─────────────────────────────────────────────────────────┐
│ for agent in [Planner, Critic, Refiner, Judger]:         │
│                                                          │
│   1. 构建消息                                            │
│      messages = build_agent_messages_sequential_text_mas │
│                 (role=agent.role, context=contexts)       │
│                                                          │
│   2. 生成文本                                            │
│      generated_texts = model.generate(messages)           │
│                                                          │
│   3. 更新上下文                                           │
│      if agent.role != "judger":                          │
│          contexts[idx] += formatted_output  # 累积        │
│      else:                                               │
│          final_texts[idx] = text_out  # Judger 输出     │
│                                                          │
│   4. 记录轨迹                                            │
│      agent_traces[idx].append(agent_trace)               │
└─────────────────────────────────────────────────────────┘

最终：final_texts = [Judger输出1, Judger输出2, ...]
```

---

## 10.5 上下文累积机制

### 10.5.1 上下文更新

```python
# 每个 Agent （非 Judger）执行后，累积上下文
if agent.role != "judger":
    contexts[idx] = f"{contexts[idx]}{formatted_output}"
```

### 10.5.2 累积过程示例

**输入问题**：
```
Question: 小明有 5 个苹果，又买了 3 个，他有多少苹果？
```

**执行过程**：

```
Step 1: Planner
contexts[0] = ""  →  ""
output: "[Planner]:\n- Step 1: ...\n- Step 2: ...\n\n"
contexts[0] = "[Planner]:\n- Step 1: ...\n- Step 2: ...\n\n"

Step 2: Critic
contexts[0] = "[Planner]:\n- Step 1: ...\n- Step 2: ...\n\n"
output: "[Critic]:\nOriginal Plan: ...\nFeedback: ...\n\n"
contexts[0] = "[Planner]:\n...\n[Critic]:\n...\n\n"

Step 3: Refiner
contexts[0] = "[Planner]:\n...\n[Critic]:\n...\n\n"
output: "[Refiner]:\nRefined Plan: ...\n\n"
contexts[0] = "[Planner]:\n...\n[Critic]:\n...\n[Refiner]:\n...\n\n"

Step 4: Judger
contexts[0] = "[Planner]:\n...\n[Critic]:\n...\n[Refiner]:\n...\n\n"
output: "The answer is \boxed{8}."
final_texts[0] = "The answer is \boxed{8}."
```

### 10.5.3 关键点

| 要点 | 说明 |
|------|------|
| **Planner → Critic** | Critic 能看到 Planner 的计划 |
| **Critic → Refiner** | Refiner 能看到 Planner 的计划和 Critic 的反馈 |
| **Refiner → Judger** | Judger 能看到完整的协作过程 |

---

## 10.6 Sequential vs Hierarchical

### 10.6.1 Sequential 架构

```python
# 顺序架构消息构建
batch_messages = [
    build_agent_messages_sequential_text_mas(
        role=agent.role,
        question=item["question"],
        context=contexts[idx],
        ...
    )
    for idx, item in enumerate(items)
]
```

**特点**：
- 每个 Agent 有明确的协作职责
- 通过 `context` 参数传递历史信息
- `formatted_output = f"[{agent.name}]:\n{text_out}\n\n"`

### 10.6.2 Hierarchical 架构

```python
# 层级架构消息构建
batch_messages = [
    build_agent_messages_hierarchical_text_mas(
        role=agent.role,
        question=item["question"],
        context=contexts[idx],
        ...
    )
    for idx, item in enumerate(items)
]
```

**特点**：
- 每个 Agent 扮演不同"专家"
- Agent 名称映射：
  ```python
  agent_name_map_for_prompt_hierarchical = {
      "Planner": "Math Agent",
      "Critic": "Science Agent",
      "Refiner": "Code Agent",
      "Judger": "Task Summrizer",
  }
  ```
- `formatted_output = f"[{agent_name_map[agent.name]}]:\n{text_out}\n\n"`

---

## 10.7 Agent 轨迹记录

### 10.7.1 轨迹结构

```python
agent_traces[idx].append(
    {
        "name": agent.name,           # "Planner", "Critic", ...
        "role": agent.role,           # "planner", "critic", ...
        "input": prompts[idx],        # 完整输入 prompt
        "input_ids": trimmed_ids,     # token ID 列表
        "input_tokens": tokens_batch[idx],  # token 字符串列表
        "output": text_out,           # Agent 输出
    }
)
```

### 10.7.2 轨迹用途

- **调试**：查看每个 Agent 的输入输出
- **分析**：理解 Agent 协作过程
- **可视化**：展示完整的推理链路

---

## 10.8 结果构建

### 10.8.1 后处理

```python
for idx, item in enumerate(items):
    final_text = final_texts[idx]

    # 根据任务类型提取答案
    if self.task in ['mbppplus', 'humanevalplus']:
        pred = extract_markdown_python_block(final_text)
        ok, error_msg = run_with_timeout(code, timeout=10)

    elif self.task in ["aime2024", "aime2025"]:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        ok = (int(pred) == int(gold))

    else:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        ok = (pred == gold)
```

### 10.8.2 完整结果

```python
results.append(
    {
        "question": item["question"],
        "gold": gold,
        "solution": item["solution"],
        "context": history_contexts[idx],    # 完整历史上下文
        "prediction": pred,                   # 预测答案
        "raw_prediction": final_text,         # Judger 原始输出
        "agents": agent_traces[idx],          # Agent 轨迹列表
        "correct": ok,                        # 是否正确
    }
)
```

---

## 10.9 TextMAS vs Baseline 对比

### 10.9.1 代码结构对比

| 对比项 | Baseline | TextMAS |
|--------|----------|---------|
| Agent 数量 | 1 | 4 |
| Agent 循环 | 无 | `for agent in self.agents` |
| 上下文 | 无 | `contexts` 累积 |
| 协作 | 无 | Planner → Critic → Refiner → Judger |

### 10.9.2 Token 消耗对比

```
Baseline：
输入 → 输出（一次性）
Token: ~100

TextMAS：
输入 → Planner → Critic → Refiner → Judger → 输出
Token: ~100 + ~50 + ~50 + ~50 + ~100 = ~350
```

### 10.9.3 效果对比

| 方法 | 准确率 | Token 消耗 | 速度 |
|------|--------|------------|------|
| Baseline | 基线 | 100% | 1x |
| TextMAS | 通常更高 | 200-400% | 较慢 |

---

## 10.10 实验：运行 TextMAS

### 10.10.1 运行命令

```bash
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --prompt sequential \
    --max_samples 5 \
    --max_new_tokens 1024
```

### 10.10.2 输出示例

```
==================== Problem #1 ====================
Question:
James has 5 marbles...

----- Agent: Planner (planner) -----
[To Tokenize]
[Planner 的完整输入，包含问题]

[Output]
[Planner 的输出：计划步骤]
---------------------------------------------
----- Agent: Critic (critic) -----
[To Tokenize]
[Critic 的输入，包含问题 + Planner 输出]

[Output]
[Critic 的输出：评估 + 反馈]
---------------------------------------------
----- Agent: Refiner (refiner) -----
[To Tokenize]
[Refiner 的输入，包含问题 + Planner + Critic 输出]

[Output]
[Refiner 的输出：改进后的计划]
---------------------------------------------
----- Agent: Judger (judger) -----
[To Tokenize]
[Judger 的输入]

[Output]
[The answer is \boxed{8}.]
---------------------------------------------
Result: Pred=8 | Gold=8 | OK=True
```

---

## 10.11 常见问题

### Q1：Token 消耗太大？

**原因**：4 个 Agent 串行生成，上下文累积。

**解决方案**：
```bash
# 限制上下文长度
--text_mas_context_length 2000

# 或使用 LatentMAS（Token 消耗减少 50-80%）
--method latent_mas
```

### Q2：Agent 输出格式不统一？

**原因**：模型没有严格遵循提示词格式。

**解决方案**：
1. 在提示词中强调格式
2. 调整 `temperature` 参数
3. 使用后处理正则提取

### Q3：为什么 Judger 不累积上下文？

**原因**：Judger 是最后一个 Agent，直接输出最终答案，不需要传递给下一个 Agent。

---

## 10.12 本课小结

### 核心要点

| 要点 | 内容 |
|------|------|
| **TextMASMethod** | 4 Agent 文本协作方法 |
| **Agent 循环** | `for agent in self.agents` |
| **上下文累积** | `contexts[idx] += formatted_output` |
| **Sequential** | 串行协作，强调反馈 |
| **Hierarchical** | 分层专家，各自独立 |

### TextMAS 执行流程

```
run_batch(items):
    │
    ├─── 初始化 contexts, history_contexts, agent_traces
    │
    ├─── for agent in [Planner, Critic, Refiner, Judger]:
    │       │
    │       ├─── build_agent_messages_sequential/hierarchical_text_mas()
    │       ├─── model.generate_text_batch()
    │       ├─── contexts[idx] += formatted_output
    │       └─── agent_traces[idx].append(trace)
    │
    └─── 后处理
            ├─── extract_gsm8k_answer() / extract_markdown_python_block()
            └─── 构建结果
```

---

## 10.13 课后练习

1. **阅读代码**：仔细阅读 `text_mas.py` 的完整实现
2. **运行实验**：运行 TextMAS 并观察 Agent 协作过程
3. **对比架构**：尝试 `--prompt sequential` 和 `--prompt hierarchical` 的差异
4. **思考问题**：
   - TextMAS 的主要优势是什么？
   - Token 消耗大的问题如何解决？

---

## 下节课预告

> **第 11 课：LatentMAS 核心原理**
> - 为什么要在潜在空间做推理？
> - Token 消耗对比：TextMAS vs LatentMAS
> - KV Cache 在潜在推理中的作用
> - 潜在空间重对齐（latent realignment）机制

---

*本教程由 Claude Code 生成，保存于 `./tutorial/` 目录*
