# 第 9 课：基线方法（baseline.py）

> 本节课时长：约 45-60 分钟
> 学习目标：理解 BaselineMethod 的实现原理，掌握单 Agent 直接生成的流程

---

## 9.1 回顾与引入

前几节课我们学习了提示词模块和消息构建函数。本节课我们将学习**三种推理方法中的第一种：基线方法（baseline）**。

### 三种方法对比

| 方法 | 文件 | Agent 数量 | 协作方式 |
|------|------|------------|----------|
| **baseline** | `baseline.py` | 1 | 无协作，直接生成 |
| **text_mas** | `text_mas.py` | 4 | 文本传递协作 |
| **latent_mas** | `latent_mas.py` | 4 | 潜在空间协作 |

### 本课学习路径

```
BaselineMethod
    │
    ├─── __init__ 初始化
    │
    ├─── run_batch 批量处理
    │
    ├─── run_item 单条处理
    │
    └─── 答案提取与评估
```

---

## 9.2 BaselineMethod 类概述

### 9.2.1 类定义

```python
class BaselineMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args=None,
    ) -> None:
        ...
```

### 9.2.2 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `self.model` | ModelWrapper | 模型封装实例 |
| `self.max_new_tokens` | int | 最大生成长度 |
| `self.temperature` | float | 采样温度 |
| `self.top_p` | float | Nucleus 采样 |
| `self.generate_bs` | int | 生成批次大小 |
| `self.use_vllm` | bool | 是否使用 vLLM |
| `self.task` | str | 任务类型 |

### 9.2.3 方法一览

| 方法 | 作用 |
|------|------|
| `run_batch(items)` | 批量处理，返回结果列表 |
| `run_item(item)` | 单条处理，调用 `run_batch([item])[0]` |

---

## 9.3 run_batch() 详解

### 9.3.1 函数签名

```python
def run_batch(self, items: List[Dict]) -> List[Dict]:
```

**输入**：`items` 是数据字典列表，每项包含：
```python
{
    "question": "问题文本",
    "solution": "解答过程",
    "gold": "标准答案"
}
```

**输出**：结果字典列表，每项包含：
```python
{
    "question": "问题文本",
    "gold": "标准答案",
    "solution": "解答过程",
    "prediction": "预测答案",
    "raw_prediction": "原始输出",
    "agents": [...],  # Agent 轨迹
    "correct": True/False
}
```

### 9.3.2 流程图

```
run_batch() 流程：

1. 构建消息
   batch_messages = [
       build_agent_messages_single_agent(question=item["question"], ...)
       for item in items
   ]

2. 准备输入张量
   prompts, input_ids, attention_mask, tokens_batch = model.prepare_chat_batch(batch_messages)

3. 批量生成
   │
   ├─── use_vllm=True  → vllm_generate_text_batch(prompts, ...)
   └─── use_vllm=False → generate_text_batch(input_ids, attention_mask, ...)

4. 后处理（每条结果）
   │
   ├─── 提取答案
   │       ├─── 代码任务：extract_markdown_python_block()
   │       ├─── AIME 任务：extract_gsm8k_answer() → int
   │       └─── 其他任务：extract_gsm8k_answer()
   │
   ├─── 比对答案
   │
   └─── 构建结果
```

---

## 9.4 构建单 Agent 消息

### 9.4.1 代码

```python
batch_messages = [
    build_agent_messages_single_agent(question=item["question"], args=self.args)
    for item in items
]
```

### 9.4.2 消息格式

```python
# build_agent_messages_single_agent 返回：
[
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud..."},
    {"role": "user", "content": "You are a helpful assistant...\nQuestion: {question}\n..."}
]
```

### 9.4.3 批量准备

```python
prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
    batch_messages, add_generation_prompt=True
)
```

**返回值**：
- `prompts`: 渲染后的文本列表
- `input_ids`: token ID 张量 [batch, seq_len]
- `attention_mask`: 注意力掩码 [batch, seq_len]
- `tokens_batch`: token 字符串列表

---

## 9.5 批量生成

### 9.5.1 分支逻辑

```python
if self.use_vllm:
    generated_batch = self.model.vllm_generate_text_batch(
        prompts,
        max_new_tokens=self.max_new_tokens,
        temperature=self.temperature,
        top_p=self.top_p,
    )
else:
    generated_batch, _ = self.model.generate_text_batch(
        input_ids,
        attention_mask,
        max_new_tokens=self.max_new_tokens,
        temperature=self.temperature,
        top_p=self.top_p,
    )
```

### 9.5.2 两种后端对比

| 对比项 | transformers | vLLM |
|--------|-------------|------|
| 输入格式 | `input_ids` | `prompts` |
| 返回值 | `(texts, past_key_values)` | `texts` |
| 速度 | 较慢 | 较快 |

---

## 9.6 答案提取与评估

### 9.6.1 分任务处理

```python
if self.task in ['mbppplus', 'humanevalplus']:
    # 代码任务：提取代码块并执行
    pred = extract_markdown_python_block(generated_text)
    ok, error_msg = run_with_timeout(code, timeout=10)

elif self.task in ["aime2024", "aime2025"]:
    # AIME 任务：提取答案并转为整数
    pred = extract_gsm8k_answer(generated_text)
    ok = (int(pred) == int(gold))

else:
    # 其他任务：标准化后比对
    pred = normalize_answer(extract_gsm8k_answer(generated_text))
    ok = (pred == gold)
```

### 9.6.2 代码任务特殊处理

```python
if self.task in ['mbppplus', 'humanevalplus']:
    pred = extract_markdown_python_block(generated_text)

    if pred is None:
        ok = False
        error_msg = "No python code block found"
    else:
        # 拼接代码 + 测试用例
        python_code_to_exe = pred + "\n" + gold
        # 执行（带超时）
        ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
```

**关键**：代码任务需要将生成的代码与测试用例拼接后执行。

### 9.6.3 AIME 任务特殊处理

```python
elif self.task in ["aime2024", "aime2025"]:
    pred = normalize_answer(extract_gsm8k_answer(generated_text))
    gold = str(item.get("gold", "")).strip()

    try:
        pred_int = int(pred)
        gold_int = int(gold)
        ok = (pred_int == gold_int)
    except ValueError:
        ok = False
```

**关键**：AIME 答案必须是整数。

---

## 9.7 构建结果

### 9.7.1 Agent 轨迹

```python
agent_trace = {
    "name": "SingleAgent",
    "role": "singleagent",
    "input": prompts[idx],
    "input_ids": trimmed_ids,
    "input_tokens": tokens_batch[idx],
    "output": generated_text,
}
```

**字段说明**：
| 字段 | 说明 |
|------|------|
| `name` | Agent 名称 |
| `role` | Agent 角色 |
| `input` | 渲染后的输入 prompt |
| `input_ids` | token ID 列表 |
| `input_tokens` | token 字符串列表 |
| `output` | 模型生成的原始输出 |

### 9.7.2 完整结果

```python
results.append(
    {
        "question": item["question"],      # 问题
        "gold": gold,                      # 标准答案
        "solution": item["solution"],       # 解答过程
        "prediction": pred,                # 预测答案
        "raw_prediction": generated_text,  # 原始输出
        "agents": [agent_trace],           # Agent 轨迹
        "correct": ok,                     # 是否正确
    }
)
```

---

## 9.8 run_item() 单条处理

### 9.8.1 代码

```python
def run_item(self, item: Dict) -> Dict:
    return self.run_batch([item])[0]
```

### 9.8.2 用途

单条处理是批量处理的包装，方便调试和测试：
```python
# 单条测试
result = baseline_method.run_item(item)

# 等价于
result = baseline_method.run_batch([item])[0]
```

---

## 9.9 与其他方法的对比

### 9.9.1 代码行数对比

| 方法 | 文件 | 代码行数 | 复杂度 |
|------|------|----------|--------|
| baseline | `baseline.py` | ~120 | 低 |
| text_mas | `text_mas.py` | ~200+ | 中 |
| latent_mas | `latent_mas.py` | ~400+ | 高 |

### 9.9.2 Token 消耗对比

```
Baseline (单 Agent)：
问题 → [Agent] → 答案
       (一次性生成)

TextMAS (4 Agent)：
问题 → [Planner] → [Critic] → [Refiner] → [Judger] → 答案
       (4次生成，Token 累积)

LatentMAS (4 Agent)：
问题 → [Planner] → [Critic] → [Refiner] → [Judger] → 答案
       (4次生成，但 Token 消耗大幅减少)
```

### 9.9.3 效果对比

| 方法 | 准确率 | Token 消耗 | 速度 |
|------|--------|------------|------|
| baseline | 基线 | 100% | 1x |
| text_mas | 通常更高 | 200-400% | 较慢 |
| latent_mas | 通常更高 | 50-80% | 较快 |

---

## 9.10 实验：运行 Baseline

### 9.10.1 运行命令

```bash
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --max_samples 10 \
    --max_new_tokens 1024
```

### 9.10.2 输出示例

```
==================== Problem #1 ====================
Question:
James has 5 marbles. He buys 3 more. How many marbles does he have now?

----- Agent: SingleAgent (singleagent) -----
[To Tokenize]
<|user|>
You are a helpful assistant...
Question: James has 5 marbles...
[/To Tokenize]

[Output]
James starts with 5 marbles. He buys 3 more, so now he has 5 + 3 = 8 marbles.
The answer is \boxed{8}.
---------------------------------------------
Result: Pred=8 | Gold=8 | OK=True

==================== Problem #2 ====================
...
```

### 9.10.3 最终输出

```json
{
  "method": "baseline",
  "model": "Qwen/Qwen3-4B",
  "accuracy": 0.8,
  "correct": 8,
  "total_time_sec": 45.67,
  "time_per_sample_sec": 4.57
}
```

---

## 9.11 常见问题

### Q1：baseline 效果太差怎么办？

**原因**：单 Agent 推理能力有限。

**解决方案**：
1. 使用更大的模型（如 `Qwen3-14B`）
2. 调整 `temperature` 和 `top_p`
3. 使用多 Agent 方法（text_mas / latent_mas）

### Q2：代码任务执行失败？

**可能原因**：
1. 生成的代码有语法错误
2. 缺少必要的 import
3. 执行超时

**解决方案**：
```python
# 查看错误信息
print(error_msg)

# 调整超时时间
--timeout 30
```

### Q3：结果中 `prediction` 为 None？

**原因**：
1. 模型没有生成 `\boxed{}` 格式
2. 答案提取正则匹配失败

**解决方案**：
```python
# 查看原始输出
print(result["raw_prediction"])

# 调整提取逻辑
```

---

## 9.12 本课小结

### 核心要点

| 要点 | 内容 |
|------|------|
| **BaselineMethod** | 单 Agent 直接生成的简单方法 |
| **run_batch()** | 批量处理的核心方法 |
| **任务适配** | 不同任务使用不同的答案提取方式 |
| **Agent 轨迹** | 记录输入、输出用于调试 |

### BaselineMethod 流程

```
run_batch(items):
    │
    ├─── build_agent_messages_single_agent()  → 构建消息
    ├─── prepare_chat_batch()                → 准备张量
    ├─── generate_text_batch() / vllm_*()    → 批量生成
    │
    └─── 后处理
            ├─── 代码任务：extract_markdown_python_block() + run_with_timeout()
            ├─── AIME 任务：extract_gsm8k_answer() → int
            └─── 其他任务：extract_gsm8k_answer()
```

---

## 9.13 课后练习

1. **阅读代码**：仔细阅读 `baseline.py` 的完整实现
2. **运行实验**：运行 baseline 并观察输出
3. **对比任务**：尝试不同任务（gsm8k、mbppplus、arc_easy）
4. **思考问题**：
   - baseline 的优势和劣势是什么？
   - 为什么代码任务需要特殊处理？

---

## 下节课预告

> **第 10 课：TextMAS 文本多智能体**
> - TextMASMethod 的实现
> - Agent 执行顺序：Planner → Critic → Refiner → Judger
> - 上下文累积（contexts）机制
> - 两种架构：sequential vs hierarchical

---

*本教程由 Claude Code 生成，保存于 `./tutorial/` 目录*
