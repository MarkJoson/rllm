# `ppo_max_token_len_per_gpu` 参数详解

## 一句话定义

> **`ppo_max_token_len_per_gpu`** 是在 **`use_dynamic_bsz=True`（动态批大小模式）** 下，每块 GPU 在一次 PPO 前/反向传播中**最多可以处理的 token 总数**。它是 dynamic batch size 的核心预算参数，控制 micro-batch 的切分粒度。

---

## 生效前提：`use_dynamic_bsz`

这个参数只在 `use_dynamic_bsz: true` 时才真正起作用：

- **`use_dynamic_bsz: false`  (默认)**
  - 忽略 `ppo_max_token_len_per_gpu`
  - 改用 `ppo_micro_batch_size_per_gpu` 固定切分

- **`use_dynamic_bsz: true`**
  - `ppo_max_token_len_per_gpu` 成为唯一的 micro-batch 预算
  - `ppo_micro_batch_size_per_gpu` 不再使用

---

## 对训练（PPO update）的影响

在 actor 的 [update_policy()](file:///home/robomaster/Research/KernelGYM/drkernel/verl_patch/workers/code/actor/dp_actor.py#445-682) 里（如 [dp_actor.py](file:///home/robomaster/Research/KernelGYM/drkernel/verl_patch/workers/code/actor/dp_actor.py) 中）：

```python
elif self.config.use_dynamic_bsz:
    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
```

`rearrange_micro_batches` 会**按序列长度将样本打包**，把总 token 数不超过 `max_token_len` 的若干样本装进同一个 micro-batch。这样做的核心好处是：

| 对比维度 | 固定 micro-batch size | 动态 token budget (`use_dynamic_bsz: true`) |
|---|---|---|
| **短序列** | 一批只有几个样本，GPU 利用率低 | 多个短序列打包进同一批，算力不浪费 |
| **长序列** | 没有保护，可能 OOM | 自动缩减每批样本数，不超 token 预算 |
| **梯度累积步数** | 固定 | 动态（取决于打包结果） |

### 直接对训练的影响：

1. **太小**：每次 forward + backward 的样本极少，gradient accumulation 步骤多，training step 变慢，吞吐下降。
2. **太大**：micro-batch 里 token 数超出显存，导致 **OOM**。
3. **合理值**：约等于 `n_samples_per_gpu × (max_prompt_length + max_response_length)`。
   - 例如：每卡 4 个样本, prompt=512, response=1024 $\to$ $4 \times 1536 = 6144$
   - 实际留出余量，常取 2 倍左右，如 `12288` 到 `16384`。

---

## 对 log prob 计算（ref policy / rollout）的影响

在配置系统（如 [_generated_agent_ppo_trainer.yaml](file:///home/robomaster/Research/rllm/rllm/trainer/config/_generated_agent_ppo_trainer.yaml)）中，通常会有如下映射：

```yaml
ref:
  log_prob_max_token_len_per_gpu: ${oc.select:actor_rollout_ref.actor.ppo_max_token_len_per_gpu, 16384}

rollout:
  log_prob_max_token_len_per_gpu: ${oc.select:actor_rollout_ref.actor.ppo_max_token_len_per_gpu, 16384}
```

这意味着 **ref policy 和 rollout engine 在计算 log prob 时会继承同一个值**（除非在各自节点单独覆盖），其影响范围包括：

1. **Ref policy log prob**（用于 PPO 的 KL penalty 计算）：控制 reference model 进行 inference-only forward pass 时的 micro-batch 大小。
2. **Rollout log prob**（用于 multi-turn 或 IS 修正等需要重新打 log prob 的场景）：控制 rollout worker 的推理批次。

---

## Sequence Parallel 的修正

当启用 Ulysses 序列并行（`ulysses_sequence_parallel_size > 1`）时，实际 token 预算会**乘以并行度**：

```python
max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
```

这是因为长序列被切片分发到了多个 GPU 上。每个 GPU 物理上只看到 $\frac{1}{\text{sp\_size}}$ 的 token。因此，为了让每个物理 GPU 处理的数量恰好达到预算，总长度阈值需要放大 `sp_size` 倍。

---

## 典型场景下的推荐值

对于诸如 **KernelGYM** 等涉及较长上下文的任务（代码生成），通常会设置得较大，例如：

```bash
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000
```

你可以参考以下公式确认该配置是否合理，从而在利用率和显存安全之间取得平衡：

$$
\text{推荐值} \approx \lceil \frac{\text{train\_batch\_size}}{\text{n\_gpus}} \rceil \times (\text{max\_prompt\_length} + \text{max\_response\_length}) \pm \text{20\% 安全余量}
$$

例如：如果 `batch=64`, 共有 8 卡 GPU, `prompt=1024`, `response=4096`
- 每卡平均样本数 $= \frac{64}{8} = 8$
- 均摊单样本 token $= 1024 + 4096 = 5120$
- 总 token $= 8 \times 5120 = 40960$
- 则可以安全地将其设置为 `40000` 到 `48000` 左右。

> **调优建议**：如果经常抛出 OOM，请直接调低该值；如果用 Profiler 查看发现 GPU 大量处于 idle（显存未用满），则适当提高该值。
