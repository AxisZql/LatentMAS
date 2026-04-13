# 第 8 课：Agent 消息构建函数

> 本节课时长：约 45-60 分钟
> 学习目标：掌握消息构建函数的调用方式，理解单 Agent 与多 Agent 的区别

---

## 8.1 回顾与引入

上节课我们学习了 `prompts.py` 的基本结构，理解了 4 种 Agent 角色和提示词模板。本节课我们将深入学习**消息构建函数**的使用方式。

### 本课学习路径

```
消息构建函数
    │
    ├─── 调用方式
    │
    ├─── 顺序架构函数
    │
    ├─── 层级架构函数
    │
    └─── 单 Agent 函数
```

---

## 8.2 消息构建函数调用方式

### 8.2.1 函数签名

所有消息构建函数都有相似的签名：

```python
def build_agent_message_xxx(
    role: str,           # Agent 角色
    question: str,        # 问题文本
    context: str = "",    # 上下文（可选）
    method = None,        # 方法名
    args = None           # 命令行参数
) -> List[Dict]:          # 返回消息列表
```

### 8.2.2 返回值

返回标准的聊天消息格式：

```python
return [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt},
]
```

### 8.2.3 调用示例

```python
# 构建 Planner 的消息
messages = build_agent_message_sequential_latent_mas(
    role="planner",
    question="小明有5个苹果，又买了3个，他有多少苹果？",
    context="",
    method="latent_mas",
    args=args
)

# messages 现在是：
# [
#     {"role": "system", "content": "You are Qwen..."},
#     {"role": "user", "content": "You are a Planner Agent...\nQuestion: ..."}
# ]
```

---

## 8.3 顺序架构函数

### 8.3.1 build_agent_message_sequential_latent_mas()

**用途**：LatentMAS 顺序架构的消息构建

**代码位置**：`prompts.py` 第 2-115 行

**核心逻辑**：

```python
def build_agent_message_sequential_latent_mas(role, question, context="", method=None, args=None):
    system_message = "You are Qwen, created by Alibaba Cloud..."

    if role == "planner":
        user_prompt = f"""You are a Planner Agent...
Question: {question}
..."""
    elif role == "critic":
        user_prompt = f"""...
Question: {question}
The plan information is provided in latent KV representation format.
..."""
    elif role == "refiner":
        user_prompt = f"""...
Question: {question}
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.
..."""
    elif role == "judger":
        # 根据任务类型选择不同的提示词
        if args.task in ['gsm8k', 'aime2024', 'aime2025']:
            user_prompt = f"""...
You must reason step-by-step and output the final answer inside \boxed{{YOUR_FINAL_ANSWER}}.
"""
        elif args.task in ["arc_easy", "arc_challenge", "gpqa", 'medqa']:
            user_prompt = f"""...
Your final answer must be selected from A,B,C,D.
For example \boxed{{A}}.
"""
        elif args.task in ["mbppplus", "humanevalplus"]:
            user_prompt = f"""...
You must put all python code in markdown code blocks.
"""
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
```

### 8.3.2 build_agent_messages_sequential_text_mas()

**用途**：TextMAS 顺序架构的消息构建

**与 LatentMAS 的区别**：

| 对比项 | LatentMAS | TextMAS |
|--------|-----------|---------|
| Critic 描述 | "latent KV format" | "in text format" |
| 信息传递 | 潜在向量 | 文本 |

**关键代码差异**：

```python
# LatentMAS
"The plan information is provided in latent KV representation format."

# TextMAS
"You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format."
```

---

## 8.4 层级架构函数

### 8.4.1 build_agent_message_hierarchical_latent_mas()

**用途**：LatentMAS 层级架构的消息构建

**代码位置**：`prompts.py` 第 118-338 行

**核心思想**：每个 Agent 扮演不同"专家角色"

### 8.4.2 Agent 角色映射

| Agent | 层级架构角色 | 描述 |
|-------|-------------|------|
| Planner | Math Agent | 数学专家，负责计算 |
| Critic | Science Agent | 科学专家，负责验证 |
| Refiner | Code Agent | 代码专家，负责实现 |
| Judger | Task Summarizer | 任务总结者，负责汇总 |

### 8.4.3 层级架构示例

**Planner（Math Agent）**：

```python
if role == "planner":
    user_content = f"""
You are a math agent. Given the input question, reason step-by-step
and put the final answer inside \boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
```

**Critic（Science Agent）**：

```python
elif role == "critic":
    user_content = f"""
You are a science agent. Given the input question, reason step-by-step
and put the final answer inside \boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
```

### 8.4.4 两种架构对比

| 架构 | 特点 | 适用场景 |
|------|------|----------|
| **Sequential** | 强调协作流程 | 复杂推理任务 |
| **Hierarchical** | 强调专家分工 | 多领域任务 |

---

## 8.5 单 Agent 函数

### 8.5.1 build_agent_messages_single_agent()

**用途**：构建单 Agent 基线方法的消息

**特点**：
- 只有一条 user message
- 直接要求模型输出答案
- 无 Agent 协作

**提示词结构**：

```
You are a helpful assistant. Given the input question, reason step-by-step
and output the final answer inside \boxed{YOUR_FINAL_ANSWER}.

Input Question: {question}

Your response:
```

### 8.5.2 与多 Agent 的对比

| 对比项 | 单 Agent | 多 Agent |
|--------|----------|----------|
| 消息数量 | 2 条（system + user）| 2 条 × Agent 数量 |
| 协作 | 无 | 有（Planner → Critic → Refiner → Judger）|
| Token 消耗 | 少 | 多 |
| 效果 | 基线 | 通常更好 |

---

## 8.6 消息构建在方法中的应用

### 8.6.1 BaselineMethod 中的使用

```python
# methods/baseline.py
from prompts import build_agent_messages_single_agent

class BaselineMethod:
    def run_batch(self, batch):
        for item in batch:
            question = item["question"]
            # 构建单 Agent 消息
            messages = build_agent_messages_single_agent(
                role="",  # 单 Agent 不需要 role
                question=question,
                context="",
                method="baseline",
                args=self.args
            )
            # 生成答案
            output = self.model.generate(messages)
```

### 8.6.2 TextMASMethod 中的使用

```python
# methods/text_mas.py
from prompts import build_agent_messages_sequential_text_mas

class TextMASMethod:
    def run_batch(self, batch):
        for item in batch:
            question = item["question"]
            contexts = []

            # Planner
            messages = build_agent_messages_sequential_text_mas(
                role="planner",
                question=question,
                context="",
                method="text_mas",
                args=self.args
            )
            planner_output = self.model.generate(messages)
            contexts.append(planner_output)

            # Critic
            messages = build_agent_messages_sequential_text_mas(
                role="critic",
                question=question,
                context=planner_output,  # 传递 Planner 输出
                method="text_mas",
                args=self.args
            )
            ...
```

### 8.6.3 LatentMASMethod 中的使用

```python
# methods/latent_mas.py
from prompts import build_agent_message_sequential_latent_mas

class LatentMASMethod:
    def run_batch(self, batch):
        for item in batch:
            question = item["question"]

            # Planner（正常文本生成）
            messages = build_agent_message_sequential_latent_mas(
                role="planner",
                question=question,
                context="",
                method="latent_mas",
                args=self.args
            )
            planner_output = self.model.generate_text(messages)

            # Critic（潜在空间生成）
            messages = build_agent_message_sequential_latent_mas(
                role="critic",
                question=question,
                context="",  # LatentMAS 不传文本上下文
                method="latent_mas",
                args=self.args
            )
            # 潜在向量会通过 KV Cache 传递
            critic_past = self.model.generate_latent(messages)

            ...
```

---

## 8.7 Context 参数的作用

### 8.7.1 TextMAS 中的 Context

```python
def build_agent_messages_sequential_text_mas(role, question, context="", ...):
    ctx = context[: args.text_mas_context_length]  # 截断上下文

    if role == "critic":
        user_content = f"""...
Plan from Planner Agent:
{ctx}  # ← 注入 Planner 的输出
..."""
```

**作用**：将上一个 Agent 的输出传递给下一个 Agent。

### 8.7.2 LatentMAS 中的 Context

```python
def build_agent_message_sequential_latent_mas(role, question, context="", ...):
    # LatentMAS 模式下，context 参数被忽略
    # 信息通过 KV Cache 传递，而非文本
    ...
```

**作用**：LatentMAS 不需要文本 context，信息在潜在空间传递。

---

## 8.8 text_mas_context_length 参数

### 8.8.1 作用

限制 TextMAS 中上下文的最大长度：

```python
ctx = context[: args.text_mas_context_length]
```

### 8.8.2 默认值

```python
# run.py
parser.add_argument("--text_mas_context_length", type=int, default=-1)
```

- `-1`：不限制
- 正整数：限制最大 token 数

### 8.8.3 为什么需要限制？

随着 Agent 数量增加，上下文会累积膨胀：
```
Planner 输出: 100 tokens
Critic 输出: 100 tokens
Refiner 输出: 100 tokens
→ 总计: 300 tokens
```

限制长度可以控制 Token 消耗。

---

## 8.9 实战：修改提示词观察效果

### 8.9.1 修改 Planner 提示词

**目标**：让 Planner 输出更详细的步骤

**原始提示词**：
```
Your outlined plan should be concise with a few bulletpoints for each step.
```

**修改后**：
```
Your outlined plan should be detailed with 3-5 bulletpoints for each step.
Include intermediate calculations in your plan.
```

### 8.9.2 修改 Judger 答案格式

**目标**：添加"推理过程"要求

**原始提示词**：
```
You must reason step-by-step and output the final answer inside \boxed{YOUR_FINAL_ANSWER}.
```

**修改后**：
```
You must first show your reasoning process, then output the final answer
inside \boxed{YOUR_FINAL_ANSWER}. Your reasoning should be between
<REASONING> and </REASONING> tags.
```

### 8.9.3 测试步骤

```bash
# 1. 修改 prompts.py
# 2. 运行实验
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --max_samples 5

# 3. 观察输出变化
```

---

## 8.10 常见问题

### Q1：提示词修改后没效果？

**可能原因**：
1. 修改后没有重新运行
2. 模型没有遵循格式要求
3. 提示词被其他设置覆盖

**解决方案**：
```python
# 确认修改已生效
print(messages)

# 增大 temperature 强制探索
--temperature 0.8
```

### Q2：Agent 输出格式不统一？

**原因**：模型没有严格遵循格式要求。

**解决方案**：
1. 在提示词中强调格式
2. 使用后处理正则提取答案
3. 调整 `temperature` 参数

### Q3：Token 消耗太大？

**原因**：提示词太长或上下文累积。

**解决方案**：
1. 简化提示词
2. 限制 `text_mas_context_length`
3. 使用 LatentMAS（Token 消耗更少）

---

## 8.11 本课小结

### 核心要点

| 要点 | 内容 |
|------|------|
| **函数签名** | `(role, question, context, method, args)` |
| **返回值** | `[{"role": "system", ...}, {"role": "user", ...}]` |
| **Sequential** | 串行协作，上下文累积 |
| **Hierarchical** | 专家分工，各自独立 |
| **Context** | TextMAS 传递文本，LatentMAS 忽略 |

### 函数对比

| 函数 | 架构 | 方法 |
|------|------|------|
| `build_agent_message_sequential_latent_mas` | Sequential | LatentMAS |
| `build_agent_message_hierarchical_latent_mas` | Hierarchical | LatentMAS |
| `build_agent_messages_sequential_text_mas` | Sequential | TextMAS |
| `build_agent_messages_hierarchical_text_mas` | Hierarchical | TextMAS |
| `build_agent_messages_single_agent` | N/A | Baseline |

---

## 8.12 课后练习

1. **阅读代码**：阅读 `prompts.py` 中的 `build_agent_messages_sequential_text_mas()` 函数
2. **对比差异**：对比 Sequential 和 Hierarchical 架构的提示词差异
3. **修改测试**：修改 Planner 的提示词，运行实验对比效果
4. **思考问题**：
   - TextMAS 和 LatentMAS 的提示词主要差异是什么？
   - 为什么 LatentMAS 不需要传递文本上下文？

---

## 下节课预告

> **第 9 课：基线方法（baseline.py）**
> - BaselineMethod 的实现原理
> - 单 Agent 直接生成流程
> - 对比实验的设计
> - 实验：运行 baseline 并分析输出

---

*本教程由 Claude Code 生成，保存于 `./tutorial/` 目录*
