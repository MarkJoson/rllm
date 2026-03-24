# Chat 模板解析器 (Chat Template Parsers)

Chat 模板解析器负责将 OpenAI-compatible 消息列表转换为模型可理解的 prompt 字符串，并将模型输出解析回结构化的 `content`、`reasoning`、`tool_calls`。

> 源码参考：[rllm/parser/](../rllm/parser/)

## ChatTemplateParser 接口

```python
class ChatTemplateParser:
    def parse(self, messages: list[dict], add_generation_prompt: bool = True,
              is_first_msg: bool = True, tools: list = [], 
              accumulate_reasoning: bool = False) -> str:
        """将消息列表转为 prompt 字符串"""

    def parse_completion(self, completion_ids: list[int]) -> dict:
        """解析 completion token IDs → {content, reasoning, tool_calls}"""

    def tokenize_and_mask(self, messages: list[dict]) -> tuple:
        """将消息 tokenize 并生成 loss mask (prompt, response, mask)"""

    def tokenize_and_mask_cumulative(self, messages: list[dict]) -> tuple:
        """多步累积消息的 tokenize + mask"""

    @classmethod
    def get_parser(cls, tokenizer, processor=None, 
                   disable_thinking=False) -> "ChatTemplateParser":
        """根据 tokenizer 自动选择合适的 parser"""
```

## 自动选择机制

`ChatTemplateParser.get_parser()` 检测 tokenizer 名称/类型，自动选择对应的 parser：

| 模型家族 | Parser 类 | 特殊处理 |
|----------|-----------|---------|
| Qwen3 | `Qwen3ChatParser` | `<think>` 推理标签、工具调用 |
| Qwen2.5 | `Qwen25ChatParser` | 工具调用格式 |
| Qwen2VL/3VL | `QwenVLChatParser` | 多模态 + mrope |
| Llama 3 | `Llama3ChatParser` | `<|eot_id|>` 分隔 |
| DeepSeek | `DeepSeekChatParser` | `<think>` 推理 |
| 其他 | `DefaultChatParser` | `tokenizer.apply_chat_template()` |

## Thinking Token 处理

`disable_thinking` 参数控制推理标签行为：

| `disable_thinking` | 行为 |
|----|------|
| `False` (默认) | `<think>` 内容保留在 response 中，用于训练 |
| `True` | `<think>...</think>` 内容被移除，不参与训练 |

## parse_completion 输出

```python
{
    "content": "The answer is 42",          # 实际内容（去除 <think>）
    "reasoning": "Let me think step by step...",  # 推理过程
    "tool_calls": [                          # 解析出的工具调用
        ToolCall(name="python", arguments={"code": "2+3"})
    ]
}
```

## tokenize_and_mask 在训练中的角色

```
messages:
  [system: "Be helpful"]           → prompt_ids (不计算 loss)
  [user: "2+3=?"]                  → prompt_ids (不计算 loss)
  [assistant: "The answer is 5"]    → response_ids, mask=1 (计算 loss)

tokenize_and_mask(messages) →
  prompt = tensor([10, 20, 30, ...])       # system + user tokens
  response = tensor([40, 50, 60, ...])     # assistant tokens
  mask = tensor([1, 1, 1, ...])            # 全为 1

tokenize_and_mask_cumulative(messages) →  # 多步时
  prompt = tensor([...initial...])
  response = tensor([asst_1, env_2, asst_2])
  mask = tensor([1,1,1, 0,0, 1,1,1])       # env tokens masked
```

## accumulate_reasoning 模式

当 `accumulate_reasoning=True` 时，前一步的 `<think>` 推理内容被保留在下一步的 prompt 中。这允许模型在多步对话中看到自己之前的思考过程。

默认为 `False`——每步 prompt 只包含 content，不包含 reasoning。
