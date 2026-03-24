# Rejection Sampling 与过滤 (Rejection Sampling and Filtering)

详见 [31-Distributed-Training-with-Ray.md](31-Distributed-Training-with-Ray.md) 中的 Rejection Sampling 部分。

---

# 蒸馏系统 (Distillation System)

详见 [31-Distributed-Training-with-Ray.md](31-Distributed-Training-with-Ray.md) 中的蒸馏系统部分。

rLLM 提供 `DistillationWorkflow` 内置工作流，支持 on-policy 蒸馏——使用当前模型的正确 rollout 作为蒸馏数据。

## 关键实现

```python
class DistillationWorkflow(Workflow):
    async def run(self, task, uid, **kwargs):
        # 1. 构建 prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task["question"]},
        ]
        
        # 2. 生成 response
        model_output = await self.rollout_engine.get_model_response(messages)
        
        # 3. 计算奖励
        reward = self.reward_fn(task, model_output.content)
        
        # 4. 构建 Step
        step = Step.from_model_output(model_output, messages=messages)
        step.reward = reward.reward
        
        # 5. 建 Episode（保留 chat_completions 供后续 SFT）
        trajectory = Trajectory(steps=[step], reward=reward.reward)
        episode = Episode(
            trajectories=[trajectory],
            is_correct=reward.is_correct,
        )
        return episode
```
