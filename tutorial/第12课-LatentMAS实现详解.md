# 第 12 课：LatentMAS 实现详解

> 本节课时长：约 60-90 分钟
> 学习目标：深入理解 LatentMAS 的代码实现，掌握关键方法的细节

---

## 12.1 回顾与引入

上节课我们学习了 LatentMAS 的核心原理。本节课我们将深入代码实现，理解 `generate_latent_batch()`、`_truncate_past()` 等核心方法的细节。

### 本课学习路径

```
LatentMASMethod
    │
    ├─── __init__ 初始化
    │
    ├─── generate_latent_batch() 潜在生成
    │
    ├─── _truncate_past() KV Cache 裁剪
    │
    ├─── run_batch() HF 后端
    │
    └─── run_batch_vllm() vLLM 后端
```

---

## 12.2 类的初始化

### 12.2.1 __init__ 代码

```python
class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps          # 潜在推理步数
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        # latent_only 时自动设置 sequential_info_only
        if self.latent_only:
            self.sequential_info_only = True

        # vLLM 采样参数
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task
```

### 12.2.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `latent_steps` | 10 | 潜在推理步数 |
| `judger_max_new_tokens` | 256 | Judger 最大生成长度 |
| `latent_only` | False | 只使用潜在推理（不生成文本）|
| `sequential_info_only` | False | 只保留最后一步的信息 |

---

## 12.3 KV Cache 裁剪

### 12.3.1 _slice_tensor() 静态方法

```python
@staticmethod
def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    if tokens_to_keep <= 0:
        return tensor[..., 0:0, :].contiguous()
    keep = min(tokens_to_keep, tensor.shape[-2])
    start = tensor.shape[-2] - keep
    return tensor[..., start:, :].contiguous()
```

**作用**：从 tensor 的末尾保留指定数量的 token。

**参数**：
- `tensor`: [batch, seq_len, hidden_dim]
- `tokens_to_keep`: 要保留的 token 数量

**示例**：
```
原始 tensor: shape = [2, 100, 4096]  (100 tokens)
tokens_to_keep = 10
结果: shape = [2, 10, 4096]  (保留最后 10 tokens)
```

### 12.3.2 _truncate_past() 方法

```python
def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
    if past_kv is None or tokens_to_keep <= 0:
        return None

    # 处理 transformers 4.x 的新 Cache 格式
    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
        trimmed_legacy = tuple(
            tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
            for layer in legacy
        )
        return past_kv.__class__.from_legacy_cache(trimmed_legacy)

    # 处理旧格式 tuple
    trimmed_layers = []
    for layer in past_kv:
        if isinstance(layer, tuple):
            trimmed_layers.append(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
            )
        elif torch.is_tensor(layer):
            trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
        else:
            trimmed_layers.append(layer)

    return tuple(trimmed_layers)
```

**作用**：裁剪 KV Cache，保留最近的 `tokens_to_keep` 个 token。

**为什么需要裁剪？**
- 潜在推理会产生大量中间状态
- KV Cache 会不断累积
- 裁剪可以控制显存占用

---

## 12.4 generate_latent_batch() 详解

### 12.4.1 代码

```python
@torch.no_grad()
def generate_latent_batch(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    latent_steps: int,
    past_key_values: Optional[Tuple] = None,
) -> Tuple:
    # 1. 处理 attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=self.device)
    else:
        attention_mask = attention_mask.to(self.device)

    # 2. 合并历史 mask
    if past_key_values is not None:
        past_len = _past_length(past_key_values)
        if past_len > 0:
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

    # 3. 初始前向传播
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values

    # 4. 获取初始隐藏状态
    e_t = outputs.hidden_states[0][:, -1, :]          # 第一层的最后 token
    last_hidden = outputs.hidden_states[-1][:, -1, :]  # 最后一层的最后 token
    h_t = last_hidden.detach().clone()

    e_t_plus_1 = None
    latent_vecs_all: List[torch.Tensor] = []
    latent_vecs_all.append(e_t.detach().clone())

    # 5. 潜在推理循环
    for step in range(latent_steps):
        # 应用重对齐
        source_model = self.HF_model if hasattr(self, "HF_model") else self.model
        latent_vec = self._apply_latent_realignment(last_hidden, source_model)

        latent_vecs_all.append(latent_vec.detach().clone())

        if step == 0:
            e_t_plus_1 = latent_vec.detach().clone()

        # 用潜在向量作为输入
        latent_embed = latent_vec.unsqueeze(1)

        # 更新 attention mask
        past_len = _past_length(past)
        latent_mask = torch.ones(
            (latent_embed.shape[0], past_len + 1),
            dtype=torch.long,
            device=self.device,
        )

        # 潜在空间前向传播
        outputs = self.model(
            inputs_embeds=latent_embed,
            attention_mask=latent_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

    return past
```

### 12.4.2 关键步骤

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 处理 attention_mask | 确保 mask 正确 |
| 2 | 合并历史 mask | 与历史 KV Cache 合并 |
| 3 | 初始前向传播 | 获取初始隐藏状态 |
| 4 | 提取隐藏状态 | `e_t` 和 `last_hidden` |
| 5 | 潜在推理循环 | 迭代 `latent_steps` 次 |

### 12.4.3 潜在推理循环详解

```python
for step in range(latent_steps):
    # 1. 潜在向量 = 重对齐(last_hidden)
    latent_vec = _apply_latent_realignment(last_hidden, model)

    # 2. 将潜在向量扩展为 [batch, 1, hidden]
    latent_embed = latent_vec.unsqueeze(1)

    # 3. 用潜在向量作为输入进行前向传播
    outputs = model(inputs_embeds=latent_embed, past_key_values=past)

    # 4. 更新 KV Cache 和隐藏状态
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]
```

---

## 12.5 run_batch() 详解

### 12.5.1 主循环结构

```python
@torch.no_grad()
def run_batch(self, items: List[Dict]) -> List[Dict]:
    batch_size = len(items)
    past_kv: Optional[Tuple] = None  # 累积的 KV Cache
    agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
    final_texts = ["" for _ in range(batch_size)]

    for agent in self.agents:
        # 1. 构建消息
        if self.args.prompt == "sequential":
            batch_messages = [
                build_agent_message_sequential_latent_mas(...)
            ]
        else:
            batch_messages = [
                build_agent_message_hierarchical_latent_mas(...)
            ]

        # 2. 准备输入
        prompts, input_ids, attention_mask, tokens_batch = model.prepare_chat_batch(...)

        if agent.role != "judger":
            # 3. 非 Judger：潜在推理
            prev_past_len = _past_length(past_kv)

            past_kv = model.generate_latent_batch(
                wrapped_ids,
                attention_mask=wrapped_mask,
                latent_steps=self.latent_steps,
                past_key_values=past_kv,
            )

            # 4. 裁剪 KV Cache
            if self.sequential_info_only or self.latent_only:
                new_past_len = _past_length(past_kv)
                tokens_added = new_past_len - prev_past_len
                tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                past_kv = self._truncate_past(past_kv, tokens_to_keep)

            # 5. 记录轨迹
            for idx in range(batch_size):
                agent_traces[idx].append({
                    "name": agent.name,
                    "role": agent.role,
                    "input": wrapped_prompts[idx],
                    "latent_steps": self.latent_steps,
                    "output": "",  # 潜在推理不输出文本
                })

        else:
            # 6. Judger：文本生成
            past_for_decoding = past_kv if self.latent_steps > 0 else None

            generated_batch, _ = model.generate_text_batch(
                judger_ids,
                judger_mask,
                max_new_tokens=self.judger_max_new_tokens,
                past_key_values=past_for_decoding,
            )

            # 7. 记录最终结果
            for idx in range(batch_size):
                final_texts[idx] = generated_batch[idx].strip()
                agent_traces[idx].append({
                    "name": agent.name,
                    "role": agent.role,
                    "output": final_texts[idx],
                })
```

### 12.5.2 流程图

```
run_batch() 执行流程：

初始化：past_kv = None

for agent in [Planner, Critic, Refiner, Judger]:
    │
    ├─── 1. 构建消息
    │
    ├─── 2. 准备输入
    │
    └─── if agent.role != "judger":
            │
            ├─── 3. generate_latent_batch() → 更新 past_kv
            ├─── 4. _truncate_past() → 裁剪
            └─── 5. 记录轨迹（output=""）
            │
    └─── else (Judger):
            │
            ├─── 6. generate_text_batch() → 生成答案
            └─── 7. 记录 final_texts

后处理：提取答案，构建结果
```

---

## 12.6 run_batch_vllm() 详解

### 12.6.1 与 run_batch() 的区别

| 对比项 | run_batch() | run_batch_vllm() |
|--------|-------------|------------------|
| 文本生成后端 | transformers | vLLM |
| 潜在推理 | HF 模型 | HF 模型 |
| 嵌入传递 | KV Cache | prompt_embeds |
| 速度 | 较慢 | 较快 |

### 12.6.2 vLLM 后端的特殊处理

```python
def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
    # ...
    for agent in self.agents:
        # ... 潜在推理部分相同 ...

        if agent.role == "judger":
            # 获取历史嵌入
            past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)

            # 获取当前 prompt 的嵌入
            curr_prompt_emb = self.model.embedding_layer(judger_ids).squeeze(0)

            # 合并嵌入
            whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)

            # 使用 vLLM 生成
            prompt_embeds_list = [{"prompt_embeds": emb} for emb in whole_prompt_emb]
            outputs = self.model.vllm_engine.generate(prompt_embeds_list, self.sampling_params)

            generated_texts = [out.outputs[0].text.strip() for out in outputs]
```

### 12.6.3 嵌入位置处理

```python
# 找到 user prompt 的插入位置
len_of_left = []
for p in judger_prompts:
    idx = p.find("<|im_start|>user\n")
    left = p[: idx + len("<|im_start|>user\n")]
    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))

# 合并嵌入
whole_prompt_emb_list = []
for i in range(B):
    insert_idx = len_of_left[i]
    left_emb = curr_prompt_emb[i, :insert_idx, :]
    right_emb = curr_prompt_emb[i, insert_idx:, :]
    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
    whole_prompt_emb_list.append(combined)
```

---

## 12.7 潜在推理参数控制

### 12.7.1 latent_only 模式

```python
if self.latent_only:
    # 只保留最后一步的潜在信息
    tokens_to_keep = self.latent_steps
    past_kv = self._truncate_past(past_kv, tokens_to_keep)
```

**用途**：减少信息量，提高专注度。

### 12.7.2 sequential_info_only 模式

```python
if self.sequential_info_only:
    # 保留所有步骤的信息
    tokens_to_keep = tokens_added  # 只保留新增的
    past_kv = self._truncate_past(past_kv, tokens_to_keep)
```

**用途**：平衡信息量和效率。

---

## 12.8 完整执行流程图

```
LatentMAS 完整执行流程：

run_batch(items):
    │
    ├─── 初始化
    │       past_kv = None
    │       agent_traces = [[], [], ...]
    │       final_texts = ["", "", ...]
    │
    ├─── for agent in [Planner, Critic, Refiner, Judger]:
    │       │
    │       ├─── 构建消息
    │       ├─── 准备输入
    │       │
    │       └─── if agent.role != "judger":
    │               │
    │               ├─── generate_latent_batch()
    │               │       │
    │               │       └─── 潜在推理循环 (latent_steps 次)
    │               │               │
    │               │               ├─── _apply_latent_realignment()
    │               │               └─── 更新 past_kv
    │               │
    │               ├─── _truncate_past()
    │               └─── 记录轨迹
    │               │
    │               └─── else (Judger):
    │                       │
    │                       ├─── generate_text_batch(past_kv)
    │                       └─── 记录 final_texts
    │
    └─── 后处理
            ├─── extract_gsm8k_answer()
            ├─── 比对 gold
            └─── 构建结果
```

---

## 12.9 常见问题与解决方案

### Q1：LatentMAS 效果不如 TextMAS？

**可能原因**：
1. `latent_steps` 设置不当
2. 模型不支持潜在推理
3. 潜在空间重对齐出问题

**解决方案**：
```bash
# 调整 latent_steps
--latent_steps 20

# 尝试禁用重对齐
--latent_space_realign
```

### Q2：显存不足？

**解决方案**：
```bash
# 减小 batch size
--generate_bs 5

# 减小 latent_steps
--latent_steps 5
```

### Q3：vLLM 后端生成失败？

**可能原因**：
- prompt_embeds 格式不对
- 嵌入维度不匹配

**解决方案**：
```bash
# 使用 HF 后端
python run.py --method latent_mas ...  # 默认使用 HF
```

---

## 12.10 本课小结

### 核心方法

| 方法 | 作用 |
|------|------|
| `generate_latent_batch()` | 潜在空间批量生成 |
| `_slice_tensor()` | 裁剪 tensor 保留末尾 token |
| `_truncate_past()` | 裁剪 KV Cache |
| `run_batch()` | HF 后端主循环 |
| `run_batch_vllm()` | vLLM 后端主循环 |

### 潜在推理核心

```python
for step in range(latent_steps):
    # 1. 重对齐隐藏状态
    latent_vec = _apply_latent_realignment(last_hidden, model)

    # 2. 潜在向量前向传播
    outputs = model(inputs_embeds=latent_vec.unsqueeze(1), past_key_values=past)

    # 3. 更新状态
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]
```

---

## 12.11 课后练习

1. **阅读代码**：仔细阅读 `latent_mas.py` 的 `generate_latent_batch()` 方法
2. **调试运行**：使用 `--max_samples 1` 运行 LatentMAS，观察 KV Cache 的变化
3. **参数调优**：尝试不同的 `latent_steps` 值
4. **思考问题**：
   - 为什么需要裁剪 KV Cache？
   - vLLM 后端和 HF 后端在潜在推理上有什么差异？

---

## 下节课预告

> **第 13 课：主程序入口（run.py）**
> - 命令行参数解析（argparse）
> - 参数详解
> - 实验配置与运行
> - 如何阅读他人代码的主入口

---

*本教程由 Claude Code 生成，保存于 `./tutorial/` 目录*
