# 第 6 课：vLLM 加速推理

> 本节课时长：约 45-60 分钟
> 学习目标：理解 vLLM 的核心优势，掌握混合后端模式的使用

---

## 6.1 回顾与引入

前两节课我们学习了 `ModelWrapper` 类的核心方法，包括 `generate_text_batch()` 文本生成、`prepare_chat_input()` 输入处理等。本节课我们将学习 **vLLM 加速推理**，这是项目高效运行的关键技术之一。

### 本课学习路径

```
为什么需要 vLLM？
    │
    ├─── transformers 的局限性
    │
    ├─── vLLM 的核心优势
    │
    └─── 混合后端模式
```

---

## 6.2 为什么需要 vLLM？

### 6.2.1 transformers 的局限性

使用纯 transformers 后端时，存在以下问题：

| 问题 | 说明 | 影响 |
|------|------|------|
| **显存占用高** | 每个请求预分配完整上下文长度 | 资源浪费 |
| **吞吐量大降** | 逐个请求处理 | GPU 利用率低 |
| **长上下文受限** | 显存不足限制上下文长度 | 无法处理长文本 |

### 6.2.2 瓶颈分析

```
transformers 推理：

请求 1: [A, B, C] → GPU 处理 → 等待
请求 2: [D, E, F] → GPU 处理 → 等待  ← 串行处理，GPU 空闲
请求 3: [G, H, I] → GPU 处理 → 等待
```

**GPU 利用率低**：大量时间花费在等待和内存分配上。

---

## 6.3 vLLM 核心优势

### 6.3.1 PagedAttention：虚拟显存管理

**核心创新**：将 KV Cache 管理从连续显存改为**分页管理**。

**传统方式**：
```
显存分配（假设 max_new_tokens = 2048）：
Token 1-100:  [████████████████████] 已使用
Token 101-2048: [                    ] 预分配但未使用 ← 浪费
```

**PagedAttention 方式**：
```
显存分配（按需分配）：
Page 1: [████████████████████] 已使用
Page 2: [████████████        ] 部分使用
Page 3: (未分配) → 按需分配
```

**优势**：
| 对比项 | 传统方式 | PagedAttention |
|--------|----------|----------------|
| 显存浪费 | ~50% | ~5% |
| 吞吐 | 低 | 高 2-10 倍 |
| 最大上下文 | 4K-8K | 16K-32K+ |

### 6.3.2 Continuous Batching：动态批处理

**问题**：固定 batch 大小导致资源利用率低。

**Continuous Batching 解决方案**：

```
时间 →
t=1: [请求A] [请求B] [请求C]      ← 3 个请求同时处理
t=2: [请求A✅] [请求B] [请求C]    ← 请求 A 完成，新请求 D 插入
t=3: [请求D] [请求B✅] [请求C✅]  ← B, C 完成
t=4: [请求D] [请求E] [请求F]      ← 新请求继续
```

**优势**：GPU 始终保持高利用率。

### 6.3.3 吞吐量对比

| 后端 | 吞吐量 (req/s) | 显存利用率 |
|------|----------------|------------|
| transformers | ~5 | ~40% |
| vLLM | ~30-50 | ~90% |

**实测提升**：3-10 倍加速

---

## 6.4 vLLM 后端使用

### 6.4.1 启用 vLLM

```bash
# 安装 vLLM
pip install vllm

# 运行实验
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --task gsm8k \
    --use_vllm \
    --max_new_tokens 2048
```

### 6.4.2 vllm_generate_text_batch()

```python
def vllm_generate_text_batch(
    self,
    prompts: List[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    if not self.vllm_engine:
        raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")

    # vLLM 采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    # 直接传入文本 prompt
    outputs = self.vllm_engine.generate(prompts, sampling_params)

    # 提取生成文本
    generations = [out.outputs[0].text.strip() for out in outputs]
    return generations
```

### 6.4.3 与 transformers 后端的区别

| 对比项 | transformers | vLLM |
|--------|--------------|------|
| 输入格式 | `input_ids` 张量 | `prompts` 文本列表 |
| 分词 | 需要手动分词 | vLLM 自动处理 |
| KV Cache | 手动管理 | 自动管理 |
| 显存 | 预分配 | 按需分配 |
| 吞吐量 | 低 | 高 |

**关键区别**：vLLM 直接接收**文本 prompt**，不需要手动分词。

---

## 6.5 vLLM 配置参数

### 6.5.1 tensor_parallel_size —— 张量并行

**作用**：将模型切分到多个 GPU 上。

```python
self.vllm_engine = LLM(
    model=model_name,
    tensor_parallel_size=2,  # 使用 2 张 GPU
    ...
)
```

**使用场景**：
- 单 GPU 显存不足
- 有多张 GPU 可用

### 6.5.2 gpu_memory_utilization —— 显存利用率

**作用**：设置 vLLM 可使用的显存比例。

```python
self.vllm_engine = LLM(
    model=model_name,
    gpu_memory_utilization=0.9,  # 使用 90% 显存
    ...
)
```

**默认值**：0.9（保留 10% 显存给 KV Cache）

**注意**：值太高可能导致 OOM，太低会降低吞吐量。

### 6.5.3 enable_prefix_caching —— 前缀缓存

**作用**：缓存相同前缀的请求，加速多 Agent 场景。

```python
self.vllm_engine = LLM(
    model=model_name,
    enable_prefix_caching=True,  # 启用前缀缓存
    ...
)
```

**应用场景**：
- LatentMAS 多 Agent 协作（Agent 间共享前缀）
- 批量处理相同前缀的请求

---

## 6.6 混合后端模式（LatentMAS 专用）

### 6.6.1 为什么需要混合模式？

LatentMAS 的核心创新是**潜在空间推理**，但 vLLM 官方不支持修改 KV Cache 或通过潜在嵌入提示。

**解决方案**：混合使用
- **vLLM**：负责最终文本生成（高效）
- **HuggingFace**：负责潜在空间推理（灵活）

### 6.6.2 架构图

```
混合后端架构：

GPU 0 (vLLM)                    GPU 1 (HuggingFace)
┌─────────────────┐            ┌─────────────────┐
│   vLLM Engine   │            │   HF Model     │
│                 │            │                 │
│  文本生成        │◄───────────│  潜在空间推理   │
│  (prefix cache) │   latent   │  (realignment) │
│                 │   embeds   │                 │
└─────────────────┘            └─────────────────┘
```

### 6.6.3 启用混合模式

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --task gsm8k \
    --use_vllm \
    --use_second_HF_model \
    --enable_prefix_caching \
    --device2 cuda:1 \
    --max_new_tokens 2048
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `--use_vllm` | 启用 vLLM |
| `--use_second_HF_model` | 同时加载 HF 模型 |
| `--enable_prefix_caching` | 启用前缀缓存 |
| `--device2 cuda:1` | HF 模型放在第二块 GPU |

### 6.6.4 代码实现

```python
if self.use_vllm:
    # 加载 vLLM 引擎
    self.vllm_engine = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
        enable_prefix_caching=True,
        enable_prompt_embeds=True,  # 支持潜在嵌入
    )

    use_second_hf = bool(getattr(args, "use_second_HF_model", False))
    if use_second_hf:
        # 加载 HuggingFace 模型（潜在推理用）
        self.HF_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(args.device2).eval()

        # 获取 embedding 层
        self.embedding_layer = self.HF_model.get_input_embeddings()
        self.HF_device = args.device2
```

---

## 6.7 潜在空间重对齐矩阵

### 6.7.1 为什么需要重对齐？

在混合模式中：
1. HF 模型输出隐藏状态 `h`（潜在向量）
2. 需要将 `h` 投影回嵌入空间
3. vLLM 才能继续生成

### 6.7.2 矩阵构建

```python
def _build_latent_realign_matrix(self, model, device, args):
    # 获取输入和输出嵌入
    input_embeds = model.get_input_embeddings()   # [vocab, hidden]
    output_embeds = model.get_output_embeddings()  # [hidden, vocab]

    input_weight = input_embeds.weight.detach()   # [vocab, hidden]
    output_weight = output_embeds.weight.detach()  # [vocab, hidden]

    # Gram 矩阵 + 正则化
    gram = torch.matmul(output_weight.T, output_weight)  # [hidden, hidden]
    gram = gram + 1e-5 * torch.eye(gram.shape[0])

    # 求解线性系统
    rhs = torch.matmul(output_weight.T, input_weight)   # [hidden, vocab]
    realign_matrix = torch.linalg.solve(gram, rhs)      # [hidden, vocab]

    # 计算目标范数
    target_norm = input_weight.norm(dim=1).mean()

    return realign_matrix, target_norm
```

### 6.7.3 数学原理

**问题**：已知输出嵌入 `W_out` 和输入嵌入 `W_in`，求变换矩阵 `M` 使得：

```
h @ M ≈ W_in  (h 是来自 W_out 的隐藏状态)
```

**解法**：通过 Gram 矩阵求逆

```
M = (W_out^T @ W_out + λI)^(-1) @ W_out^T @ W_in
```

---

## 6.8 完整执行流程

### 6.8.1 LatentMAS 混合模式流程

```
1. Planner Agent
   ├── HF_model.generate() → 生成计划
   └── 提取 hidden_states

2. Critic Agent (潜在空间)
   ├── _apply_latent_realignment(hidden) → latent_vec
   ├── HF_model(inputs_embeds=latent_vec) → 更新 past_key_values
   └── 生成批评

3. Refiner Agent (潜在空间)
   ├── _apply_latent_realignment(hidden) → latent_vec
   ├── HF_model(inputs_embeds=latent_vec) → 更新 past_key_values
   └── 生成改进

4. Judger Agent (vLLM)
   ├── past_key_values → 嵌入向量
   └── vllm_engine.generate() → 最终答案
```

### 6.8.2 效率对比

| 模式 | Token 消耗 | 速度 |
|------|-----------|------|
| TextMAS (transformers) | 100% | 1x |
| LatentMAS (transformers) | 20-50% | 3-7x |
| LatentMAS (vLLM) | 20-50% | 10-20x |

---

## 6.9 常见问题与解决方案

### Q1：vLLM 报 CUDA 版本不匹配？

**错误**：`RuntimeError: CUDA version mismatch`

**解决方案**：
```bash
# 检查 CUDA 版本
nvcc --version

# 重新安装匹配版本的 vLLM
pip install vllm --index-url https://wheels.example.com/cu118
```

### Q2：混合模式显存不足？

**解决方案**：
1. 减小 `gpu_memory_utilization`：
   ```bash
   --gpu_memory_utilization 0.7
   ```
2. 使用更小的模型：
   ```bash
   --model_name Qwen/Qwen3-4B
   ```
3. 减小 batch size：
   ```bash
   --generate_bs 10
   ```

### Q3：vLLM 输出为空？

**可能原因**：
- `enable_prefix_caching=True` 但 prompt 格式不对
- `enable_prompt_embeds=True` 但未正确传入嵌入

**解决方案**：
```bash
# 禁用前缀缓存
python run.py --method latent_mas ... --enable_prefix_caching
```

---

## 6.10 本课小结

### 核心要点

| 要点 | 内容 |
|------|------|
| **PagedAttention** | 分页式 KV Cache 管理，减少显存浪费 |
| **Continuous Batching** | 动态批处理，提高 GPU 利用率 |
| **tensor_parallel_size** | 张量并行，多 GPU 加速 |
| **gpu_memory_utilization** | 显存利用率控制 |
| **enable_prefix_caching** | 前缀缓存，多 Agent 共享 |
| **混合后端模式** | vLLM 生成 + HF 潜在推理 |

### vLLM 配置对比

| 场景 | 配置 |
|------|------|
| 单 GPU，baseline | `use_vllm=True` |
| 多 GPU，LatentMAS | `use_vllm=True`, `use_second_HF_model=True` |
| 长上下文 | `gpu_memory_utilization=0.85` |
| 高吞吐 | `enable_prefix_caching=True` |

---

## 6.11 课后练习

1. **对比测试**：分别用 transformers 和 vLLM 运行 baseline，对比速度
2. **阅读代码**：查看 `vllm_generate_text_batch()` 的完整实现
3. **尝试混合模式**：使用双 GPU 运行 LatentMAS 混合模式
4. **思考问题**：
   - PagedAttention 为什么能减少显存占用？
   - 混合模式下 HF 模型和 vLLM 如何协作？

---

## 下节课预告

> **第 7 课：提示词模块初探（prompts.py）**
> - Agent 角色定义：Planner、Critic、Refiner、Judger
> - 默认 Agent 系统消息的结构
> - 不同任务的答案格式要求
> - 如何阅读复杂的提示词模板

---

*本教程由 Claude Code 生成，保存于 `./tutorial/` 目录*
