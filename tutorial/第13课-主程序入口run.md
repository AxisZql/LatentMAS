# 第 13 课：主程序入口（run.py）

> 本节课时长：约 45-60 分钟
> 学习目标：理解 run.py 的结构，掌握参数解析和实验配置

---

## 13.1 回顾与引入

前几节课我们学习了三种推理方法的实现。本节课我们将学习整个项目的**主入口：run.py**，理解如何将各个模块串联起来。

### 本课学习路径

```
run.py 结构
    │
    ├─── 参数解析
    │
    ├─── 模型初始化
    │
    ├─── 方法选择
    │
    ├─── 数据加载
    │
    ├─── 批量处理
    │
    └─── 结果输出
```

---

## 13.2 run.py 整体结构

### 13.2.1 核心函数

| 函数 | 行号 | 作用 |
|------|------|------|
| `evaluate()` | 26 | 计算准确率和正确数 |
| `process_batch()` | 33 | 处理一个批次 |
| `main()` | 84 | 主函数 |

### 13.2.2 导入模块

```python
import argparse
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

from data import (
    load_aime2024, load_aime2025,
    load_arc_easy, load_arc_challenge,
    load_gsm8k, load_gpqa_diamond,
    load_mbppplus, load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time
```

---

## 13.3 evaluate() 函数

### 13.3.1 代码

```python
def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct
```

### 13.3.2 作用

计算预测结果的准确率。

**输入**：`preds` - 结果字典列表

```python
[
    {"question": "...", "correct": True},
    {"question": "...", "correct": False},
    ...
]
```

**输出**：`(accuracy, correct_count)`

---

## 13.4 process_batch() 函数

### 13.4.1 代码

```python
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds

    current_batch = batch[:remaining]

    if args.method == "latent_mas" and args.use_vllm:
        results = method.run_batch_vllm(current_batch)
    else:
        results = method.run_batch(current_batch)

    if len(results) > remaining:
        results = results[:remaining]

    # 处理并打印结果
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1

        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())

        agents = res.get("agents", [])
        for a in agents:
            print(f"----- Agent: {a.get('name')} ({a.get('role')}) -----")
            print("[To Tokenize]")
            print(a.get("input", "").rstrip())
            if a.get("latent_steps") is not None:
                print("[Latent Steps]")
                print(a.get("latent_steps"))
            print("[Output]")
            print(a.get("output", "").rstrip())
            print("----------------------------------------------")

        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))

    return processed, preds
```

### 13.4.2 处理流程

```
process_batch() 流程：

1. 计算剩余样本数
   remaining = max_samples - processed

2. 截断批次（如果超过剩余数量）
   current_batch = batch[:remaining]

3. 调用方法生成结果
   │
   ├─── latent_mas + vllm → run_batch_vllm()
   └─── 其他 → run_batch()

4. 打印每个样本的详细信息
   ├─── Question
   ├─── Agent 轨迹（输入、输出）
   └─── 预测结果 vs 标准答案

5. 更新进度条
   progress.update(len(results))

6. 返回更新后的状态
   return processed, preds
```

---

## 13.5 main() 函数详解

### 13.5.1 参数解析

```python
def main():
    parser = argparse.ArgumentParser()

    # ============ 核心参数 ============
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential")

    # ============ 生成参数 ============
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20)
    parser.add_argument("--text_mas_context_length", type=int, default=-1)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # ============ vLLM 参数 ============
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--use_second_HF_model", action="store_true")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    args = parser.parse_args()
```

### 13.5.2 参数分类

| 类别 | 参数 | 说明 |
|------|------|------|
| **核心** | `--method`, `--model_name`, `--task`, `--prompt` | 基本配置 |
| **数据** | `--max_samples`, `--split` | 数据控制 |
| **生成** | `--max_new_tokens`, `--temperature`, `--top_p` | 生成控制 |
| **方法** | `--latent_steps`, `--latent_space_realign` | 方法参数 |
| **vLLM** | `--use_vllm`, `--device2`, `--gpu_memory_utilization` | vLLM 配置 |

---

## 13.6 模型初始化

### 13.6.1 代码

```python
# 设置随机种子
set_seed(args.seed)

# 自动选择设备
device = auto_device(args.device)

# 初始化模型
model = ModelWrapper(
    args.model_name,
    device,
    use_vllm=args.use_vllm,
    args=args
)
```

### 13.6.2 ModelWrapper 初始化参数

| 参数 | 来源 | 说明 |
|------|------|------|
| `model_name` | `--model_name` | 模型名称 |
| `device` | `auto_device(args.device)` | 计算设备 |
| `use_vllm` | `--use_vllm` | 是否使用 vLLM |
| `args` | `args` | 完整参数对象 |

---

## 13.7 方法选择

### 13.7.1 代码

```python
common_kwargs = dict(
    temperature=args.temperature,
    top_p=args.top_p,
)

if args.method == "baseline":
    method = BaselineMethod(
        model,
        max_new_tokens=args.max_new_tokens,
        **common_kwargs,
        generate_bs=args.generate_bs,
        use_vllm=args.use_vllm,
        args=args
    )
elif args.method == "text_mas":
    method = TextMASMethod(
        model,
        max_new_tokens_each=args.max_new_tokens,
        **common_kwargs,
        generate_bs=args.generate_bs,
        args=args,
    )
elif args.method == 'latent_mas':
    method = LatentMASMethod(
        model,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        **common_kwargs,
        generate_bs=args.generate_bs,
        args=args,
    )
```

### 13.7.2 三种方法的参数

| 方法 | 特有参数 | 说明 |
|------|----------|------|
| **baseline** | `max_new_tokens` | 最大生成长度 |
| **text_mas** | `max_new_tokens_each` | 每个 Agent 的最大生成长度 |
| **latent_mas** | `latent_steps`, `judger_max_new_tokens` | 潜在步数、Judger 生成长度 |

---

## 13.8 数据加载

### 13.8.1 代码

```python
if args.task == "gsm8k":
    dataset_iter = load_gsm8k(split=args.split)
elif args.task == "aime2024":
    dataset_iter = load_aime2024(split="train")  # AIME 只有 train 集
elif args.task == "aime2025":
    dataset_iter = load_aime2025(split='train')
elif args.task == "gpqa":
    dataset_iter = load_gpqa_diamond(split='test')
elif args.task == "arc_easy":
    dataset_iter = load_arc_easy(split='test')
elif args.task == "arc_challenge":
    dataset_iter = load_arc_challenge(split='test')
elif args.task == "mbppplus":
    dataset_iter = load_mbppplus(split='test')
elif args.task == "humanevalplus":
    dataset_iter = load_humanevalplus(split='test')
elif args.task == "medqa":
    dataset_iter = load_medqa(split='test')
else:
    raise ValueError(f'no {args.task} support')

# 转换为列表（如果需要全部样本）
if args.max_samples == -1:
    dataset_iter = list(dataset_iter)
    args.max_samples = len(dataset_iter)
```

### 13.8.2 数据集 split 默认值

| 数据集 | 默认 split | 备注 |
|--------|-----------|------|
| `gsm8k` | test | |
| `aime2024` | train | 只有 train 集 |
| `aime2025` | train | 只有 train 集 |
| `gpqa` | test | |
| `arc_easy` | test | |
| `arc_challenge` | test | |
| `mbppplus` | test | |
| `humanevalplus` | test | |
| `medqa` | test | |

---

## 13.9 批量处理循环

### 13.9.1 代码

```python
preds: List[Dict] = []
processed = 0
batch: List[Dict] = []

progress = tqdm(total=args.max_samples)

for item in dataset_iter:
    if processed >= args.max_samples:
        break

    batch.append(item)

    if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            args.max_samples,
            args,
        )
        batch = []

    if processed >= args.max_samples:
        break

# 处理剩余的 batch
if batch and processed < args.max_samples:
    processed, preds = process_batch(
        method,
        batch,
        processed,
        preds,
        progress,
        max_samples=args.max_samples,
        args=args,
    )

progress.close()
```

### 13.9.2 批量处理流程

```
批量处理流程：

1. 初始化
   preds = []
   processed = 0
   batch = []

2. 迭代数据集
   for item in dataset_iter:
       │
       ├─── 添加到 batch
       │       batch.append(item)
       │
       ├─── batch 满了？或到最后了？
       │       len(batch) == generate_bs or processed + len(batch) == max_samples
       │       │
       │       └─── Yes:
       │               process_batch(batch)
       │               batch = []
       │
       └─── processed >= max_samples？ → break

3. 处理剩余 batch
   if batch:
       process_batch(batch)
```

---

## 13.10 结果输出

### 13.10.1 代码

```python
total_time = time.time() - start_time

acc, correct = evaluate(preds)

print(
    json.dumps(
        {
            "method": args.method,
            "model": args.model_name,
            "split": args.split,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "accuracy": acc,
            "correct": correct,
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / args.max_samples, 4),
        },
        ensure_ascii=False,
    )
)
```

### 13.10.2 输出示例

```json
{
  "method": "latent_mas",
  "model": "Qwen/Qwen3-4B",
  "split": "test",
  "seed": 42,
  "max_samples": 100,
  "accuracy": 0.85,
  "correct": 85,
  "total_time_sec": 1234.5678,
  "time_per_sample_sec": 12.3457
}
```

---

## 13.11 完整运行流程

```
run.py 完整流程：

1. parse_args()        ← 解析命令行参数

2. set_seed()          ← 设置随机种子

3. auto_device()        ← 选择设备

4. ModelWrapper()       ← 初始化模型

5. 创建 Method          ← 根据 method 参数选择

6. 加载数据集           ← 根据 task 参数选择

7. 批量处理循环         ← process_batch()
        │
        ├─── 迭代数据集
        ├─── 积累 batch
        ├─── 调用 method.run_batch()
        └─── 打印结果

8. evaluate()           ← 计算准确率

9. 输出 JSON 结果
```

---

## 13.12 常用运行命令

### 12.1 基线方法

```bash
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --max_samples -1 \
    --max_new_tokens 2048
```

### 12.2 TextMAS 方法

```bash
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --prompt sequential \
    --max_samples -1 \
    --max_new_tokens 2048
```

### 12.3 LatentMAS 方法

```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --prompt sequential \
    --latent_steps 10 \
    --max_samples -1 \
    --max_new_tokens 2048
```

### 12.4 LatentMAS + vLLM

```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --task gsm8k \
    --use_vllm \
    --use_second_HF_model \
    --enable_prefix_caching \
    --device2 cuda:1 \
    --max_samples -1 \
    --max_new_tokens 2048
```

---

## 13.13 本课小结

### run.py 结构

| 部分 | 函数/代码 | 说明 |
|------|-----------|------|
| 辅助函数 | `evaluate()` | 计算准确率 |
| 辅助函数 | `process_batch()` | 处理批次 |
| 主函数 | `main()` | 串联整个流程 |

### main() 流程

```
main():
    │
    ├─── 1. 解析参数 (argparse)
    ├─── 2. 设置种子 (set_seed)
    ├─── 3. 选择设备 (auto_device)
    ├─── 4. 初始化模型 (ModelWrapper)
    ├─── 5. 选择方法 (BaselineMethod / TextMASMethod / LatentMASMethod)
    ├─── 6. 加载数据 (load_xxx)
    ├─── 7. 批量处理 (process_batch 循环)
    └─── 8. 输出结果 (JSON)
```

---

## 13.14 课后练习

1. **阅读代码**：仔细阅读 `run.py` 的完整实现
2. **运行实验**：使用不同的参数组合运行实验
3. **分析输出**：理解 JSON 输出格式的含义
4. **思考问题**：
   - 如何添加一个新的数据集到 `run.py`？
   - 如何添加新的命令行参数？

---

## 下节课预告

> **第 14 课：实验设计与结果分析**
> - 如何设计一组对比实验
> - 评估指标：准确率、正确数、耗时
> - JSON 结果输出格式解析
> - 绘图与可视化实验结果
> - 常见问题排查

---

*本教程由 Claude Code 生成，保存于 `./tutorial/` 目录*
