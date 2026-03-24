# Rollout 引擎 (Rollout Engines)

Rollout 引擎抽象了 LLM 推理层，将不同的推理后端（vLLM Python API、OpenAI HTTP API、Tinker HTTP、Fireworks API）统一为一致的 `ModelOutput` 接口。

> 源码参考：
> - [rllm/engine/rollout/rollout_engine.py L1-67](../rllm/engine/rollout/rollout_engine.py)
> - [rllm/engine/rollout/verl_engine.py L1-115](../rllm/engine/rollout/verl_engine.py)
> - [rllm/engine/rollout/openai_engine.py](../rllm/engine/rollout/openai_engine.py)
> - [rllm/engine/rollout/tinker_engine.py](../rllm/engine/rollout/tinker_engine.py)
> - [rllm/engine/rollout/fireworks_engine.py](../rllm/engine/rollout/fireworks_engine.py)

## 基类定义

`RolloutEngine` ([rollout_engine.py L55-67](../rllm/engine/rollout/rollout_engine.py#L55-L67))：

```python
class RolloutEngine:
    def __init__(self, *args, **kwargs): pass

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        raise NotImplementedError

    async def wake_up(self): pass    # 加载推理权重到 GPU
    async def sleep(self): pass       # 卸载推理权重
```

> `wake_up()` / `sleep()` 是 verl 独有的 GPU 显存管理机制——其他引擎的实现为空操作。

## ModelOutput 数据结构

([rollout_engine.py L7-52](../rllm/engine/rollout/rollout_engine.py#L7-L52))

```python
@dataclass
class ModelOutput:
    text: str | None = None              # 完整文本输出
    content: str | None = None           # 内容部分（移除 <think>）
    reasoning: str | None = None         # 推理过程（<think> 内容）
    tool_calls: list[ToolCall] | None = None  # 解析出的工具调用
    prompt_ids: list[int] | None = None  # prompt token IDs
    completion_ids: list[int] | None = None  # completion token IDs
    multi_modal_inputs: dict | None = None  # VLM 多模态输入
    logprobs: list[float] | None = None  # completion log probs
    prompt_logprobs: list[float] | None = None  # prompt log probs
    prompt_length: int = 0
    completion_length: int = 0
    finish_reason: str | None = None     # "stop" | "length"
```

**序列化**：`to_dict()` 和 `from_dict()` 支持 JSON 序列化/反序列化。

## VerlEngine：vLLM Python API

([verl_engine.py L12-115](../rllm/engine/rollout/verl_engine.py#L12-L115))

### 初始化

```python
class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        # 验证 rollout 引擎类型
        assert config.actor_rollout_ref.rollout.name in ["vllm", "sglang"]

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, server_handles=...)
        self.tokenizer = tokenizer
        self.processor = processor  # VLM 处理器（Qwen2VL 等）
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, ...)

        # 训练/验证采样参数
        self.train_sampling_params = dict(
            temperature=config...temperature,
            top_k=config...top_k, top_p=config...top_p,
            logprobs=1,  # 始终收集 logprobs
        )
        self.val_sampling_params = dict(...)  # 验证用温度通常为 0
```

### get_model_response 流程

([verl_engine.py L48-106](../rllm/engine/rollout/verl_engine.py#L48-L106))

```
1. 参数处理 [L49-60]
   ├── 提取 application_id, validate, tools, accumulate_reasoning
   └── 选择 train/val sampling_params

2. 模板解析 [L62-63]
   prompt = chat_parser.parse(messages, add_generation_prompt=True, tools=tools)
   request_prompt_ids = tokenizer.encode(prompt)

3. VLM 多模态处理 [L65-74]  
   if messages 包含 images 且有 processor:
       image_data = chat_parser.process_image_data(messages)
       model_inputs = processor(text=[prompt], images=image_data)
       prompt_ids = model_inputs.pop("input_ids")[0]
       multi_modal_inputs = dict(model_inputs)  # image_grid_thw 等

4. 长度检查 [L77-78]
   if enforce_max_prompt_length and len(prompt_ids) > max_prompt_length:
       raise TerminationEvent(MAX_PROMPT_LENGTH_EXCEEDED)

5. vLLM 推理 [L80-82]
   token_output = await server_manager.generate(
       request_id=application_id,
       prompt_ids=request_prompt_ids,
       image_data=image_data,
       sampling_params=sampling_params,
   )
   completion_ids = token_output.token_ids
   logprobs = token_output.log_probs

6. 截断处理 [L84-88]
   if len(completion_ids) >= max_tokens:
       finish_reason = "length"
       completion_ids = completion_ids[:max_tokens]
       logprobs = logprobs[:max_tokens]

7. 输出解析 [L90-92]
   completion_text = tokenizer.decode(completion_ids)
   parsed = chat_parser.parse_completion(completion_ids)
   → {"content": ..., "reasoning": ..., "tool_calls": ...}

8. 构建 ModelOutput [L94-106]
```

### wake_up / sleep 机制

([verl_engine.py L108-114](../rllm/engine/rollout/verl_engine.py#L108-L114))

```python
async def wake_up(self):
    """并发唤醒所有 rollout replicas"""
    await asyncio.gather(*[
        replica.wake_up() for replica in self.rollout_manager.rollout_replicas
    ])

async def sleep(self):
    """并发休眠所有 rollout replicas"""
    await asyncio.gather(*[
        replica.sleep() for replica in self.rollout_manager.rollout_replicas
    ])
```

> **GPU 显存时分复用原理**：
> - `wake_up()`：每个 vLLM replica 加载模型权重到 GPU → 启动推理服务
> - `sleep()`：每个 vLLM replica 释放权重 + KV Cache → GPU 显存归还给训练
> - 训练步内：先 `wake_up()` → 生成轨迹 → `sleep()` → 训练更新

## OpenAIEngine：HTTP API

特点：
- 兼容任何 OpenAI-compatible API（包括 vLLM 的 HTTP 服务器）
- 支持超时重试（`api_retries` 参数）
- 自动从 response 头提取 prompt/completion IDs（如果后端支持）
- 支持 tool_calls 工具调用解析
- 通过 `chat_parser` 完成 prompt 模板渲染和输出解析
- 支持 `disable_thinking` 去除 `<think>` 标签

核心流程：
```python
async def get_model_response(self, messages, **kwargs):
    prompt = self.chat_parser.parse(messages, ...)
    response = await openai_client.chat.completions.create(
        model=self.model, messages=..., max_tokens=...,
        temperature=..., top_p=...,
    )
    # 解析 response → ModelOutput
    # 如果 API 不返回 prompt_ids/completion_ids → 本地 tokenize
```

## TinkerEngine：本地/服务

特点：
- 支持本地模型（直接 Python API）和远程 HTTP API
- 集成 Tinker 的微调 API
- 无需 CUDA（可在 CPU 上运行）
- Python >= 3.11

## FireworksEngine：远程 API

特点：
- 通过 Fireworks AI 的 HTTP API 进行推理
- 模型托管在远程 GPU 上
- 适合评估和小规模训练
- 无本地 GPU 需求

## 引擎对比

| 维度 | VerlEngine | OpenAIEngine | TinkerEngine | FireworksEngine |
|------|-----------|-------------|-------------|----------------|
| **推理方式** | vLLM Python API | HTTP API | 本地/HTTP | HTTP API |
| **logprobs** | ✅ 原生支持 | ✅ 如API支持 | ✅ | ✅ |
| **token IDs** | ✅ 原生返回 | ⚠️ 需本地tokenize | ✅ | ⚠️ 需本地tokenize |
| **多模态** | ✅ (Qwen2VL/3VL) | ❌ | ❌ | ❌ |
| **wake_up/sleep** | ✅ GPU时分复用 | ❌ 空操作 | ❌ 空操作 | ❌ 空操作 |
| **延迟** | 最低（进程内） | 中（HTTP） | 高（启动开销） | 高（网络） |
| **适用** | 训练循环 | 独立评估 | 开发/调试 | 远程推理 |
