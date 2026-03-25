# rLLM Framework Architecture — Component Structure, Data Flow & Agent Internals

## 1. High-Level Component Relationship

```mermaid
graph TB
    subgraph DataLayer["Data Layer"]
        DS[Dataset / DataLoader]
    end

    subgraph ExecutionLayer["Execution Layer"]
        AEE["AgentExecutionEngine<br/><small>Agent-Env loop</small>"]
        AWE["AgentWorkflowEngine<br/><small>Workflow.run()</small>"]
        ASE["AgentSdkEngine<br/><small>LiteLLM Proxy interception</small>"]
    end

    subgraph RolloutLayer["Rollout Engines"]
        RE_BASE["RolloutEngine (ABC)"]
        VE["VerlEngine<br/><small>vLLM/SGLang Python API</small>"]
        OE["OpenAIEngine<br/><small>OpenAI-compatible HTTP</small>"]
        TE["TinkerEngine<br/><small>Local/HTTP API</small>"]
        FE["FireworksEngine<br/><small>Remote HTTP API</small>"]
    end

    subgraph AgentEnvLayer["Agent & Environment"]
        BA["BaseAgent (ABC)"]
        BE["BaseEnv (ABC)"]
        WF["Workflow (ABC)"]
    end

    subgraph TransformLayer["Transform Layer"]
        TR["transform_results_for_verl()<br/><small>Episode → tokenize → pad → mask → DataProto</small>"]
    end

    subgraph TrainingLayer["Training Layer"]
        AT["AgentTrainer<br/><small>Unified entry point</small>"]
        PPO["verl PPO Trainer<br/><small>Actor / Critic / Ref</small>"]
        ADV["Advantage Estimator<br/><small>GRPO / REINFORCE / RLOO</small>"]
    end

    DS -->|"task dicts"| AEE
    DS -->|"task dicts"| AWE
    DS -->|"task dicts"| ASE

    AEE -->|"uses"| BA
    AEE -->|"uses"| BE
    AEE -->|"calls"| RE_BASE

    AWE -->|"pools"| WF
    WF -->|"owns"| BA
    WF -->|"owns"| BE
    AWE -->|"calls"| RE_BASE

    ASE -->|"intercepts via LiteLLM"| RE_BASE

    RE_BASE --- VE
    RE_BASE --- OE
    RE_BASE --- TE
    RE_BASE --- FE

    AEE -->|"list&lt;Episode&gt;"| TR
    AWE -->|"list&lt;Episode&gt;"| TR
    ASE -->|"list&lt;Episode&gt;"| TR

    TR -->|"DataProto (tensors)"| PPO
    PPO --> ADV

    AT -->|"backend=verl"| PPO
    AT -->|"backend=tinker"| TE
    AT -->|"backend=fireworks"| FE

    style DataLayer fill:#fde8ec,stroke:#e94560,color:#333
    style ExecutionLayer fill:#e8f0fd,stroke:#4a7fc1,color:#333
    style RolloutLayer fill:#eae8fd,stroke:#7c5cbf,color:#333
    style AgentEnvLayer fill:#f0e8fd,stroke:#9b5cc4,color:#333
    style TransformLayer fill:#fdeae8,stroke:#ef233c,color:#333
    style TrainingLayer fill:#e8f4fd,stroke:#2196f3,color:#333
```

### Critical Path (Training Loop)

```
Dataset → AgentWorkflowEngine.execute_tasks_verl()
       → wake_up() → Workflow.run() × N (parallel) → sleep()
       → transform_results_for_verl() → DataProto
       → Critic forward → Ref forward → Advantage → PPO update
       → checkpoint → next iteration
```

---

## 2. Type System: Step → Trajectory → Episode

```mermaid
classDiagram
    class Step {
        +list~int~ prompt_ids
        +list~int~ response_ids
        +list~float~ logprobs
        +list~dict~ chat_completions
        +Any observation
        +str thought
        +Any action
        +str model_response
        +ModelOutput model_output
        +float reward
        +bool done
        +float mc_return
        +float|list advantage
        +from_model_output(ModelOutput) Step
    }

    class Trajectory {
        +str uid
        +str name
        +dict task
        +list~Step~ steps
        +float reward
        +is_cumulative() bool
    }

    class Episode {
        +str id
        +dict task
        +bool is_correct
        +TerminationReason termination_reason
        +dict metrics
        +list~Trajectory~ trajectories
        +dict artifacts
    }

    class TrajectoryGroup {
        +list~Trajectory~ trajectories
        +str group_id
        +group_role() str
        +task_id() str
    }

    Episode "1" *-- "1..*" Trajectory : contains
    Trajectory "1" *-- "1..*" Step : contains
    TrajectoryGroup "1" o-- "1..*" Trajectory : groups for advantage
```

---

## 3. Agent Architecture Detail

### 3.1 BaseAgent Interface & Concrete Implementations

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        +chat_completions: list~dict~
        +trajectory: Trajectory
        +update_from_env(obs, reward, done, info)
        +update_from_model(response) Action
        +reset()*
        +get_current_state() Step
    }

    class ToolAgent {
        -str system_prompt
        -list tools
        -Trajectory _trajectory
        -list _messages
        +parse_action(response)
    }

    class MathAgent {
        -str system_prompt
        +format_observation()
    }

    class CodeAgent {
        -str system_prompt
        +parse_code_action()
    }

    class SWEAgent {
        -str system_prompt
        +parse_swe_command()
    }

    class WebArenaAgent {
        -str system_prompt
        +parse_browser_action()
    }

    class FrozenLakeAgent {
        +parse_direction()
    }

    class MiniWoBAgent {
        +parse_web_action()
    }

    BaseAgent <|-- ToolAgent
    BaseAgent <|-- MathAgent
    BaseAgent <|-- CodeAgent
    BaseAgent <|-- SWEAgent
    BaseAgent <|-- WebArenaAgent
    BaseAgent <|-- FrozenLakeAgent
    BaseAgent <|-- MiniWoBAgent
```

**AgentHub** (external agent integrations in `agenthub/`):

| Agent | Framework | Integration Mode |
|-------|-----------|-----------------|
| `react_agent` | Native rLLM | Agent/Env or Workflow |
| `langgraph_agent` | LangGraph | SDK Engine (LiteLLM Proxy) |
| `smolagents_agent` | SmolAgents | SDK Engine |
| `strands_agent` | Strands | SDK Engine |
| `swe_agent` | SWE-Agent | Agent/Env |
| `frozenlake_agent` | Classic RL | Agent/Env |
| `terminal_agent` | Terminal | Agent/Env |

### 3.2 BaseAgent Lifecycle — Sequence Diagram

```mermaid
sequenceDiagram
    participant Engine as AgentExecutionEngine
    participant Agent as BaseAgent
    participant Rollout as RolloutEngine
    participant Env as BaseEnv

    Note over Engine: === Episode Start ===

    Engine->>Env: env.reset()
    Env-->>Engine: (observation, info)

    Engine->>Agent: agent.reset()
    Engine->>Agent: agent.update_from_env(obs, 0.0, False, info)

    rect rgb(225, 230, 250)
        Note over Engine: Tokenize initial prompt
        Engine->>Agent: messages = agent.chat_completions
        Engine->>Engine: prompt_len = tokenize(messages)
        Engine->>Engine: if prompt_len > max_prompt_length: ABORT
    end

    loop for step_idx in range(max_steps)
        rect rgb(220, 230, 250)
            Note over Engine: 1. Prepare prompt
            Engine->>Agent: prompt = agent.chat_completions.copy()
        end

        rect rgb(230, 220, 250)
            Note over Engine: 2. LLM inference
            Engine->>Rollout: model_output = get_model_response(prompt)
            Rollout-->>Engine: ModelOutput{text, prompt_ids, completion_ids, logprobs}
        end

        rect rgb(220, 245, 230)
            Note over Engine: 3. Agent processes response
            Engine->>Agent: action = agent.update_from_model(response)
            Note right of Agent: Parse thought/action<br/>Create Step<br/>Append to trajectory
        end

        rect rgb(250, 235, 220)
            Note over Engine: 4. Environment step
            Engine->>Env: (obs, reward, done, info) = env.step(action)
            Note right of Env: Timeout guard:<br/>trajectory_timeout - elapsed
        end

        rect rgb(220, 240, 240)
            Note over Engine: 5. Update agent state
            Engine->>Agent: agent.update_from_env(obs, reward, done, info)
            Engine->>Agent: cur_step = agent.get_current_state()
            Engine->>Engine: cur_step.reward = reward, cur_step.done = done
        end

        rect rgb(250, 230, 230)
            Note over Engine: 6. Token accumulation & length check
            Engine->>Engine: response_token_len += new_tokens
            alt response_token_len >= max_response_length
                Engine->>Engine: TRUNCATION → break
            else done == True
                Engine->>Engine: ENV_DONE → break
            else timeout exceeded
                Engine->>Engine: TIMEOUT → break
            end
        end
    end

    Note over Engine: === Post-processing ===

    opt overlong_filter enabled
        Engine->>Engine: Zero out response_masks for TRUNCATION/MAX_STEPS/TIMEOUT
    end

    opt env has compute_final_reward
        Engine->>Env: reward = env.compute_final_reward()
    end

    Engine->>Env: env.close()
    Engine->>Engine: compute_trajectory_reward(trajectory)
    Engine->>Engine: compute_mc_return(trajectory, gamma)
    Engine-->>Engine: return Trajectory / TokenResult
```

### 3.3 Rollout Backend Dispatch

```mermaid
sequenceDiagram
    participant Engine as ExecutionEngine
    participant Dispatch as get_model_response()
    participant VE as VerlEngine
    participant OE as OpenAIEngine
    participant TE as TinkerEngine

    Engine->>Dispatch: get_model_response(prompt, app_id, **params)

    alt engine_name == "verl"
        Dispatch->>VE: chat_parser.parse(messages)<br/>tokenizer.encode()
        VE->>VE: server_manager.generate(prompt_ids, sampling_params)
        Note right of VE: vLLM Python API<br/>Returns token_ids + logprobs
        VE-->>Dispatch: ModelOutput{prompt_ids, completion_ids, logprobs, content}
    else engine_name == "openai"
        Dispatch->>OE: openai_client.chat.completions.create()
        Note right of OE: HTTP API<br/>+ local tokenize for IDs
        OE-->>Dispatch: ModelOutput{content, logprobs}
    else engine_name == "tinker"
        Dispatch->>TE: tinker HTTP or local API
        TE-->>Dispatch: ModelOutput
    end

    Dispatch-->>Engine: ModelOutput (unified)
```

---

## 4. Execution Engine Comparison — Sequence Diagrams

### 4.1 AgentExecutionEngine (Agent-Env Loop)

```mermaid
sequenceDiagram
    participant Trainer as PPO Trainer
    participant AEE as AgentExecutionEngine
    participant Pool as Semaphore Pool
    participant Agent as BaseAgent × N
    participant Env as BaseEnv × N
    participant RE as RolloutEngine

    Trainer->>AEE: update_envs_and_agents(envs, agents)
    Trainer->>AEE: trajectory_generator(mode="Token")

    opt engine_name == "verl"
        AEE->>RE: wake_up()
        Note right of RE: Load vLLM weights to GPU
    end

    par Parallel agent-env loops (Semaphore limited)
        AEE->>Pool: acquire semaphore slot
        Pool->>AEE: slot granted
        AEE->>Agent: run_agent_trajectory_async(idx)
        Note over Agent,Env: Full lifecycle as in §3.2
        AEE-->>Trainer: yield TokenResult
    and
        AEE->>Pool: acquire semaphore slot
        AEE->>Agent: run_agent_trajectory_async(idx+1)
        AEE-->>Trainer: yield TokenResult
    end

    opt engine_name == "verl"
        AEE->>RE: sleep()
        Note right of RE: Release GPU memory for training
    end
```

### 4.2 AgentWorkflowEngine (Custom Workflow)

```mermaid
sequenceDiagram
    participant PPO as PPO Trainer
    participant AWE as AgentWorkflowEngine
    participant WFPool as Workflow Pool (Queue)
    participant WF as Workflow Instance
    participant Solver as solver: BaseAgent
    participant Judge as judge: BaseAgent
    participant RE as RolloutEngine

    PPO->>AWE: execute_tasks_verl(batch)
    AWE->>RE: wake_up()

    AWE->>AWE: initialize_pool() [N Workflow instances]

    loop for each task
        AWE->>WFPool: workflow = queue.get()
        AWE->>WF: run_with_termination_handling(task, uid)

        WF->>WF: reset(task, uid)
        Note right of WF: Auto-reset all BaseAgent<br/>& BaseEnv attributes

        WF->>Solver: update_from_env(initial_obs)
        WF->>RE: get_model_response(solver.chat_completions)
        RE-->>WF: ModelOutput
        WF->>Solver: update_from_model(response)
        WF->>WF: commit(name="solver", agent=solver)

        WF->>Judge: update_from_env(solver_answer)
        WF->>RE: get_model_response(judge.chat_completions)
        RE-->>WF: ModelOutput
        WF->>Judge: update_from_model(response)
        WF->>WF: commit(name="judge", agent=judge)

        WF-->>AWE: Episode{trajectories=[solver_traj, judge_traj]}
        AWE->>WFPool: queue.put(workflow)
    end

    AWE->>RE: sleep()
    AWE->>AWE: transform_results_for_verl(episodes)
    AWE-->>PPO: DataProto
```

### 4.3 AgentSdkEngine (Framework-Agnostic)

```mermaid
sequenceDiagram
    participant PPO as PPO Trainer
    participant SDK as AgentSdkEngine
    participant Proxy as LiteLLM Proxy
    participant Store as SQLite Trace Store
    participant UserCode as User Agent Function<br/>(LangGraph / SmolAgents / etc.)
    participant VE as VerlEngine (vLLM)

    PPO->>SDK: execute_tasks_verl(batch)
    SDK->>VE: wake_up()

    par Parallel user functions
        SDK->>UserCode: wrapped_agent_run_func(task)
        UserCode->>Proxy: OpenAI client → POST /chat/completions
        Proxy->>VE: get_model_response(messages)
        VE-->>Proxy: ModelOutput (token_ids + logprobs)
        Proxy-->>UserCode: chat.completion response
        Proxy->>Store: record Trace (request + response + token_ids)
        UserCode-->>SDK: reward: float | list[Trajectory]
    end

    SDK->>Store: flush_traces()
    SDK->>Store: get_by_session_uid(uid, since=start_time)
    Store-->>SDK: list[Trace]

    SDK->>SDK: trace_to_step() → group_steps() → Episode
    SDK->>VE: sleep()
    SDK->>SDK: transform_results_for_verl(episodes)
    SDK-->>PPO: DataProto
```

---

## 5. End-to-End Training Data Flow

```mermaid
sequenceDiagram
    participant DS as Dataset
    participant AT as AgentTrainer
    participant Ray as Ray Cluster
    participant TaskRunner as TaskRunner
    participant Engine as Execution Engine
    participant RE as RolloutEngine
    participant Transform as Transform Layer
    participant IActor as Actor FSDP
    participant Ref as Ref Model
    participant Critic as Critic FSDP
    participant Adv as Advantage Estimator

    DS->>AT: train_dataset, val_dataset
    AT->>Ray: ray.init(runtime_env)
    AT->>TaskRunner: runner.run.remote(config, workflow_class, ...)

    loop Training Iteration
        TaskRunner->>DS: batch = next(dataloader)
        Note right of DS: DataProto with task_dicts and task_ids

        TaskRunner->>Engine: execute_tasks_verl(batch)

        rect rgb(215, 230, 250)
            Note over RE: [INFERENCE PHASE] GPU: vLLM
            Engine->>RE: wake_up() - load weights and KV cache
            Engine->>Engine: run N agent-env loops in parallel
            Engine->>RE: sleep() - release GPU memory
        end

        Engine->>Transform: list of Episodes
        Transform->>Transform: tokenize, pad, mask
        Transform-->>TaskRunner: DataProto with input_ids, attention_mask, response_mask, rewards

        rect rgb(255, 225, 225)
            Note over IActor: [TRAINING PHASE] GPU FSDP
            TaskRunner->>Ref: ref_log_probs = ref.forward(input_ids)
            TaskRunner->>IActor: actor_log_probs = actor.forward(input_ids)
            TaskRunner->>Critic: values = critic.forward(input_ids)
            TaskRunner->>Adv: advantages = compute(rewards, values)
            Note right of Adv: GRPO: group_normalize(rewards)<br/>REINFORCE: rewards - baseline<br/>RLOO: leave_one_out(rewards)
            TaskRunner->>IActor: PPO update: clip(ratio * advantages)
            TaskRunner->>Critic: value loss update
        end

        TaskRunner->>TaskRunner: save checkpoint (optional)
    end
```

---

## 6. Workflow System — Structure & Variants

```mermaid
classDiagram
    class Workflow {
        <<abstract>>
        +RolloutEngine rollout_engine
        +ThreadPoolExecutor executor
        +float timeout
        +float gamma
        +float reward_bonus_coeff
        +run(task, uid)* Episode
        +commit(name, agent, trajectory)
        +collect_trajectories() Episode
        +postprocess_episode(episode, reason) Episode
        +compute_trajectory_reward(traj)
        +adjust_step_rewards(traj)
        +reset(task, uid)
    }

    class SimpleWorkflow {
        +run(): single Q&A, no env
    }

    class SingleTurnWorkflow {
        +run(): one agent-env step
    }

    class MultiTurnWorkflow {
        +run(): multi-step agent-env
    }

    class CumulativeWorkflow {
        +run(): multi-step + token budget
    }

    class DistillationWorkflow {
        +run(): on-policy distillation
    }

    class EvalProtocolWorkflow {
        +run(): evaluation protocol
    }

    Workflow <|-- SimpleWorkflow
    Workflow <|-- SingleTurnWorkflow
    Workflow <|-- MultiTurnWorkflow
    Workflow <|-- CumulativeWorkflow
    Workflow <|-- DistillationWorkflow
    Workflow <|-- EvalProtocolWorkflow

    Workflow "1" --> "1" RolloutEngine : shared
    Workflow "1" --> "0..*" BaseAgent : auto-discovered
    Workflow "1" --> "0..*" BaseEnv : auto-discovered
```

---

## 7. Backend Architecture & Selection

```mermaid
graph LR
    subgraph Backends["Training Backends"]
        VERL["<b>VERL</b><br/>Ray + FSDP + vLLM<br/>Distributed multi-GPU<br/>GPU time-sharing"]
        TINKER["<b>Tinker</b><br/>Single-machine<br/>Local/HTTP API<br/>Dev & debug"]
        FIREWORKS["<b>Fireworks</b><br/>Remote GPU inference<br/>Pipeline training<br/>Workflow only"]
    end

    subgraph EngineSupport["Execution Engine Support"]
        E1["AgentExecutionEngine"]
        E2["AgentWorkflowEngine"]
        E3["AgentSdkEngine"]
    end

    VERL --- E1
    VERL --- E2
    VERL --- E3
    TINKER --- E1
    TINKER --- E2
    FIREWORKS --- E2

    style VERL fill:#d8f3e3,stroke:#52b788,color:#1b4332
    style TINKER fill:#e8e9f0,stroke:#81b29a,color:#3d405b
    style FIREWORKS fill:#fde8d8,stroke:#f4845f,color:#5a2d0c
```

| Dimension | VERL | Tinker | Fireworks |
|-----------|------|--------|-----------|
| **Inference** | vLLM/SGLang Python API | Local/HTTP | Remote HTTP |
| **Training** | FSDP distributed | Single-GPU gradient | Pipeline remote |
| **GPU sharing** | ✅ wake_up/sleep | N/A | N/A |
| **Distributed** | ✅ Ray | ❌ | ❌ |
| **VLM support** | ✅ Qwen2VL/3VL | Partial | ❌ |
| **Agent/Env mode** | ✅ | ✅ | ❌ |
| **Workflow mode** | ✅ | ✅ | ✅ (only) |
| **SDK mode** | ✅ | ❌ | ❌ |
| **Best for** | Production training | Development/debug | Remote inference |

---

## 8. GPU Time-Sharing Mechanism (VERL Critical Path)

```mermaid
sequenceDiagram
    participant GPU as GPU Memory
    participant vLLM as vLLM Replicas
    participant FSDP as FSDP Workers

    rect rgb(210, 240, 220)
        Note over GPU: ★ INFERENCE PHASE
        GPU->>vLLM: wake_up(): Load model weights + allocate KV cache
        vLLM->>vLLM: N concurrent agent-env trajectories
        vLLM->>vLLM: prompt → tokens + logprobs
        vLLM->>GPU: sleep(): Release weights + KV cache
    end

    rect rgb(255, 225, 225)
        Note over GPU: ★ TRAINING PHASE
        GPU->>FSDP: Load FSDP shards
        FSDP->>FSDP: Actor forward → actor_log_probs
        FSDP->>FSDP: Ref forward → ref_log_probs
        FSDP->>FSDP: Critic forward → values
        FSDP->>FSDP: Advantage computation
        FSDP->>FSDP: PPO gradient update
        FSDP->>GPU: Release shards
    end

    Note over GPU: ↻ Repeat next iteration
```

---

## 9. Component Ecological Niches — Interface & Role Reference

本节从五个核心组件的维度，分别梳理其 **主要功能**、**对外提供接口**（其他组件可调用它什么）、**对外访问接口**（它依赖哪些其他组件的接口）以及 **生态位**（在整体训练流水线中扮演的角色）。

---

### 9.1 ExecutionEngine（执行引擎）

**包含实现：** `AgentExecutionEngine` / `AgentWorkflowEngine` / `AgentSdkEngine`

#### 主要功能

- 批量并发地驱动 Agent ↔ Env 交互循环（或 Workflow / SDK 函数）。
- 管理并发度（`Semaphore` / `Queue`），限制同时活跃的轨迹数。
- 拼装多步对话 token（prompt + response 拼接、mask 标记），生成可供训练的 `TokenResult`。
- 负责 `RolloutEngine.wake_up()` / `sleep()` 的 GPU 时间片调度（仅 VERL backend）。
- 捕捉超时、截断、overlong 等异常情况并打标记。

#### 对外提供接口

| 接口 | 签名 | 调用方 |
|------|------|--------|
| `update_envs_and_agents` | `(envs, agents) → None` | Trainer（VERL PPO 轮次开始） |
| `trajectory_generator` | `async gen (mode) → TokenResult \| Trajectory` | Trainer（异步迭代获取轨迹） |
| `execute_tasks` | `async (tasks: list[dict]) → list[Trajectory]` | 独立评估 / Tinker backend |
| `execute_tasks_verl` | `(batch: DataProto) → DataProto` | AgentPPOTrainer（WorkflowEngine 版本） |
| `get_model_response` | `async (prompt, app_id, **kw) → ModelOutput` | 内部调用；Workflow 也可直接调用 |

#### 对外访问接口

| 依赖组件 | 调用的接口 |
|----------|-----------|
| `BaseAgent` | `reset()`, `update_from_env()`, `update_from_model()`, `get_current_state()`, `.chat_completions`, `.trajectory` |
| `BaseEnv` | `reset()`, `step(action)`, `close()`, `compute_final_reward()` (optional) |
| `RolloutEngine` | `get_model_response()`, `wake_up()`, `sleep()` |
| `transforms` | `transform_results_for_verl(episodes) → DataProto` |

#### 生态位

> ExecutionEngine 是 **数据生产者**。它处于 Dataset 与 Trainer 之间，将静态任务字典转化为含 token 序列、reward、response mask 的训练样本，是 agentic RL 区别于标准 RL 的核心基础设施。三种实现覆盖了"标准 Agent/Env 循环 → 自定义 Workflow → 第三方 SDK"三条路径，用户只需选择合适的一条插入自己的组件。

---

### 9.2 Agent（智能体）

**接口定义：** `BaseAgent`（`rllm/agents/agent.py`）

#### 主要功能

- 维护当前对话状态：`chat_completions`（消息列表）与 `trajectory`（步骤序列）。
- 接收环境观测并更新内部消息历史（`update_from_env`）。
- 解析模型文本响应，提取 thought/action 并写入最新 `Step`（`update_from_model`）。
- 在每个 episode 开始时清空状态（`reset`）。
- 通过 `get_current_state()` 向引擎暴露当前 `Step`，供写入 reward/done。

#### 对外提供接口

| 接口 | 签名 | 说明 |
|------|------|------|
| `reset()` | `→ None` | 清空对话历史与轨迹 |
| `update_from_env()` | `(obs, reward, done, info) → None` | 将环境反馈写入消息历史 |
| `update_from_model()` | `(response: str) → Action` | 解析响应，返回可执行 Action |
| `get_current_state()` | `→ Step \| None` | 返回轨迹最新 Step |
| `.chat_completions` | `→ list[dict]` | 当前完整对话消息（用于送入 LLM） |
| `.trajectory` | `→ Trajectory` | 所有已完成步骤的序列 |

#### 对外访问接口

Agent 本身 **不主动调用** 任何外部接口；它是纯被动的状态机，所有方法均由 ExecutionEngine 或 Workflow 驱动。

#### 生态位

> Agent 是 **状态持有者与解析器**。它封装了"如何把 LLM 文本转化为结构化动作"的领域逻辑，是唯一需要用户深度定制的组件。不同任务只需继承 `BaseAgent` 并覆写 `update_from_env` / `update_from_model`，其余训练基础设施完全复用。

```mermaid
classDiagram
    class BaseAgent {
        <<abstract>>
        +chat_completions: list[dict]
        +trajectory: Trajectory
        +reset()*
        +update_from_env(obs, reward, done, info)
        +update_from_model(response) Action
        +get_current_state() Step
    }
    note for BaseAgent "提供接口：上方全部方法\n访问接口：无（被动状态机）\n自定义点：update_from_env / update_from_model"
```

---

### 9.3 Trainer（训练器）

**主要实现：** `AgentTrainer`（`rllm/trainer/agent_trainer.py`）

#### 主要功能

- 统一用户入口：一行代码注册 `workflow_class` 或 `(agent_class, env_class)` 并绑定数据集与配置。
- 根据 `backend` 参数派发到正确的底层训练器（VERL / Tinker / Fireworks）。
- 对 VERL backend：初始化 Ray 集群，将 `TaskRunner` 作为 Ray Actor 远程运行迭代训练循环。
- 对 Tinker backend：本地调用 `TinkerAgentTrainer.fit_agent()` 做单机训练。
- 对 Fireworks backend：使用 `PipelineTaskRunner` 驱动远程推理 + 本地更新的流水线。

#### 对外提供接口

| 接口 | 签名 | 调用方 |
|------|------|--------|
| `__init__()` | `(workflow_class?, agent_class?, env_class?, config, dataset, backend)` | 用户训练脚本 |
| `train()` | `→ None` | 用户训练脚本 |

#### 对外访问接口

| 依赖组件 | 调用关系 |
|----------|---------|
| `ExecutionEngine` | 通过 `TaskRunner.run()` 间接创建并调用 `execute_tasks_verl()` / `trajectory_generator()` |
| `Dataset` | 取 `train_files` / `val_files` 路径注入到 config |
| `RolloutEngine` (verl) | 由底层 `AgentPPOTrainer` 持有，Trainer 不直接调用 |
| Ray | `ray.init()` + `runner.run.remote()` |

#### 生态位

> Trainer 是 **系统的顶层编排器**，也是用户的唯一交互入口。它对用户隐藏了 Ray、FSDP、vLLM 等的所有细节，只暴露"注册你的 Agent/Env/Workflow + 给我数据集 + 运行"这一层语义。同时它是多 backend 的统一门面，支持研究到生产的平滑迁移。

---

### 9.4 Env（环境）

**接口定义：** `BaseEnv`（`rllm/environments/base/base_env.py`）

#### 主要功能

- 实现标准 Gym 语义：`reset()` 返回初始观测，`step(action)` 返回 `(obs, reward, done, info)`。
- 提供工厂方法 `from_dict(info)` 支持从字典批量实例化（与 Dataset 的 task dict 解耦）。
- 通过 `is_multithread_safe()` 声明是否支持并发调用（AgentExecutionEngine 强制校验）。
- 可选实现 `compute_final_reward()` 延迟计算最终奖励（适合需要全局评判的任务）。

#### 对外提供接口

| 接口 | 签名 | 调用方 |
|------|------|--------|
| `reset()` | `→ (obs: dict, info: dict)` | ExecutionEngine（episode 开始） |
| `step(action)` | `→ (obs, reward: float, done: bool, info: dict)` | ExecutionEngine（每步） |
| `close()` | `→ None` | ExecutionEngine（episode 结束） |
| `compute_final_reward()` | `→ float`（可选） | ExecutionEngine（episode 结束后） |
| `from_dict(info)` | `(dict) → BaseEnv`（工厂） | ExecutionEngine.execute_tasks |
| `is_multithread_safe()` | `→ bool` | ExecutionEngine（初始化校验） |
| `.idx` | 属性 read/write | ExecutionEngine（tracking 用） |

#### 对外访问接口

Env 本身 **不调用** rLLM 框架的任何接口。它可以自由调用外部工具（代码沙箱、浏览器、数据库等），框架不约束其内部实现。

#### 生态位

> Env 是 **奖励信号的来源与任务状态机**。它定义了"什么是正确的动作、什么是任务完成"，是 RL 问题形式化的载体。框架对 Env 的唯一要求是遵守 Gym 接口，因此几乎任何外部环境（代码执行器、网页浏览器、数学验证器、游戏引擎）都可以包装为 `BaseEnv` 接入训练流程。

---

### 9.5 RolloutEngine（推理引擎）

**接口定义：** `RolloutEngine`（`rllm/engine/rollout/rollout_engine.py`）

#### 主要功能

- 提供统一的 LLM 推理接口，屏蔽底层推理服务差异（vLLM Python API / OpenAI HTTP / Tinker / Fireworks）。
- 返回标准化的 `ModelOutput`，包含文本内容、token IDs、per-token logprobs、reasoning 等训练所需全量信息。
- 通过 `wake_up()` / `sleep()` 配合 GPU 时间共享机制，在推理阶段与训练阶段之间切换 GPU 占用。
- 多实例并发安全（所有调用均通过 `async` 接口）。

#### 对外提供接口

| 接口 | 签名 | 调用方 |
|------|------|--------|
| `get_model_response()` | `async (messages: list[dict], **kw) → ModelOutput` | ExecutionEngine、Workflow |
| `wake_up()` | `async → None` | ExecutionEngine（推理阶段开始） |
| `sleep()` | `async → None` | ExecutionEngine（推理阶段结束） |

**`ModelOutput` 字段（调用方可访问）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `content` | `str` | 模型回复文本 |
| `reasoning` | `str` | 思维链内容（若有） |
| `prompt_ids` | `list[int]` | prompt token IDs |
| `completion_ids` | `list[int]` | 生成 token IDs |
| `logprobs` | `list[float]` | per-token log probabilities |
| `finish_reason` | `str` | 生成终止原因 |

#### 对外访问接口

| 依赖 | 说明 |
|------|------|
| `VerlEngine` → vLLM Server | Python API（`generate()`），需 `wake_up()` 加载权重 |
| `OpenAIEngine` → OpenAI HTTP | `openai.chat.completions.create()` |
| `TinkerEngine` → Local/HTTP | 本地推理或 Tinker HTTP API |
| `FireworksEngine` → Remote HTTP | Fireworks 远程推理 API |

#### 生态位

> RolloutEngine 是 **LLM 推理的统一抽象层**。它将 "向模型发一条消息" 这一操作与底层推理基础设施解耦，使 ExecutionEngine 和 Workflow 可以在不修改任何逻辑的情况下在 vLLM、OpenAI、远程 API 等之间切换。`wake_up/sleep` 是其独有的训练感知机制，让单 GPU 机器可以交替承担推理和训练两个阶段。

---

### 9.6 五组件交互总览

```mermaid
graph LR
    User["用户训练脚本"] -->|"train()"| Trainer

    Trainer -->|"backend=verl\nray.remote"| Engine["ExecutionEngine\n(AEE / AWE / SDK)"]

    Engine -->|"reset() / step() / close()"| Env["BaseEnv\n环境实现"]
    Engine -->|"update_from_env()\nupdate_from_model()"| Agent["BaseAgent\n智能体实现"]
    Engine -->|"get_model_response()\nwake_up() / sleep()"| Rollout["RolloutEngine\n(Verl/OpenAI/Tinker)"]

    Env -->|"(obs, reward, done, info)"| Engine
    Agent -->|"Action"| Engine
    Rollout -->|"ModelOutput"| Engine

    Engine -->|"tokenize+pad+mask\n→ DataProto"| Training["PPO Trainer\n(Actor/Critic/Ref)"]
    Training -->|"更新权重"| Rollout

    style Trainer fill:#e8f4fd,stroke:#2196f3,color:#333
    style Engine fill:#e8f0fd,stroke:#4a7fc1,color:#333
    style Env fill:#f0e8fd,stroke:#9b5cc4,color:#333
    style Agent fill:#fde8ec,stroke:#e94560,color:#333
    style Rollout fill:#eae8fd,stroke:#7c5cbf,color:#333
    style Training fill:#e8f4fd,stroke:#52b788,color:#333
```

| 组件 | 生态位关键词 | 用户自定义点 |
|------|-------------|-------------|
| **Trainer** | 顶层编排器，统一入口 | 传入 class、config、dataset |
| **ExecutionEngine** | 数据生产者，并发编排 | 通常无需修改 |
| **Agent** | 状态持有者，响应解析器 | **必须继承定制** |
| **Env** | 奖励来源，任务状态机 | **必须继承定制** |
| **RolloutEngine** | LLM 推理抽象层 | 按 backend 选择，无需直接修改 |

---

## 10. AgentPPOTrainer 训练主循环详解

`AgentPPOTrainer` 继承自 verl 的 `RayPPOTrainer`，在 `fit_agent()` 中实现了完整的 Agentic RL 训练循环。

### 10.1 训练循环全貌

```mermaid
flowchart TD
    A([开始]) --> B[加载 checkpoint\n_load_checkpoint]
    B --> C{val_before_train?}
    C --是--> D[_validate_agent\n计算初始 val 指标]
    C --否--> E
    D --> E[global_steps = 1]

    E --> F[["for epoch in epochs\nfor batch in dataloader"]]
    F --> G["DataProto = from_single_dict(batch_dict)\n分配 uid\nbatch.repeat(rollout.n, interleave=True)\n弹出 input_ids / attention_mask / position_ids"]

    G --> H[init_envs_and_agents\n并行创建 Env×N, Agent×N]

    H --> I{stepwise_advantage\n.enable?}

    I --False--> J["generate_agent_trajectory()\nmode='Token'\n每条 traj → 单条 token 序列"]
    I --True--> K["generate_agent_steps()\nmode='Step'\n每步 → 独立训练行"]

    J --> L["_transform_agent_trajectories()\n见 §11.1"]
    K --> M["_transform_agent_steps()\n见 §11.2"]

    L --> N[batch.union DataProto]
    M --> N

    N --> O{use_critic?}
    O --是--> P["critic_wg.compute_values(batch)\n→ batch['values']"]
    O --否--> Q

    P --> Q["rejection_sample + advantage\n见 §12"]

    Q --> R{use_critic?}
    R --是--> S["critic_wg.update_critic(batch)\n梯度步: 价值网络"]
    R --否--> T

    S --> T{critic_warmup\n已过?}
    T --是--> U["actor_rollout_wg.update_actor(batch)\n梯度步: PPO clip 策略更新"]
    T --否--> V

    U --> V{test_freq\n命中?}
    V --是--> W[_validate_agent]
    V --否--> X

    W --> X{save_freq\n命中?}
    X --是--> Y[_save_checkpoint]
    X --否--> Z

    Y --> Z[global_steps++\nlog metrics]
    Z --> ZZ{达到\ntotal_training_steps?}
    ZZ --否--> F
    ZZ --是--> END([结束])
```

### 10.2 关键配置与对应行为

| 配置键 | 说明 | 影响 |
|--------|------|------|
| `rllm.stepwise_advantage.enable` | 是否逐步优势 | 决定 `generate_agent_trajectory` vs `generate_agent_steps` |
| `rllm.stepwise_advantage.mode` | `broadcast` / `per_step` | 影响 advantage 归一化分组 |
| `rllm.rejection_sample.enable` | 是否启用拒绝采样 | 过滤全对/全错 episode 组 |
| `rllm.agent.overlong_filter` | 超长过滤 | 截断/超时样本的 response_mask 全零化 |
| `rllm.mask_truncated_samples` | 截断样本掩码 | 过滤最后 token 仍有效的样本（未正常结束）|
| `actor_rollout_ref.rollout.n` | 每题 rollout 数 | 每条 task 生成 n 条轨迹，用于 GRPO 分组 |
| `trainer.critic_warmup` | Critic 预热步数 | 前 N 步只更新 Critic，稳定基线 |

---

## 11. 轨迹转换双路径详解

执行引擎按 `stepwise_advantage.enable` 走两条不同的转换路径，将原始交互数据打包为训练张量。

### 11.1 _transform_agent_trajectories（默认路径）

适用于 `stepwise_advantage.enable = False`，**每条轨迹 → 1 行 DataProto**。

```mermaid
sequenceDiagram
    participant Engine as AgentExecutionEngine
    participant Trans as _transform_agent_trajectories
    participant DP as DataProto

    Engine-->>Trans: list[dict]{prompt_tokens, response_tokens,\nresponse_masks, trajectory_reward,\nchat_completions, metrics}

    Note over Trans: ① 提取各列表
    Note over Trans: ② 指标聚合 → mean/min/max
    Note over Trans: ③ 保存 chat_completions.jsonl

    rect rgb(220, 235, 255)
        Note over Trans: ④ Prompt 左填充<br/>flip → pad_sequence → flip<br/>截断: [:, -max_prompt_length:]
    end

    rect rgb(255, 225, 215)
        Note over Trans: ⑤ Response 右填充<br/>pad_sequence<br/>截断: [:, :max_response_length]
    end

    Note over Trans: ⑥ input_ids = concat(prompts, responses)
    Note over Trans: ⑦ attention_mask<br/>prompt_mask(右对齐) + response_mask(左对齐)
    Note over Trans: ⑧ loss mask = response_masks 右填充
    Note over Trans: ⑨ position_ids = cumsum(attention_mask) - 1
    Note over Trans: ⑩ 奖励置于最后有效 response token<br/>score_batch[i, resp_len-1] = reward

    Trans-->>DP: {input_ids[B,P+R], attention_mask,\nposition_ids, prompts[B,P],\nresponses[B,R], token_level_scores[B,R],\nresponse_mask[B,R]}
```

**关键设计**：
- Prompt **左填充** 使 prompt 末尾与 response 首字节相邻，保持因果注意力连续性
- 奖励置于最后 token 是 verl GRPO 的约定（`sum(scores, dim=-1)` 即标量奖励）
- `response_mask` 区分 LLM 生成 token（=1）与环境/用户 token（=0），仅对前者计算 loss

### 11.2 _transform_agent_steps（Stepwise 路径）

适用于 `stepwise_advantage.enable = True`，**每步 → 1 行 DataProto**。

```mermaid
sequenceDiagram
    participant Engine as AgentExecutionEngine
    participant Trans as _transform_agent_steps
    participant DP as DataProto

    Engine-->>Trans: list[dict]{steps[{prompt,response},...],\ntrajectory_reward, mc_returns,\ntermination_reason, idx}

    Note over Trans: ① 遍历每个 episode 的每个 step
    Note over Trans: ② 逐步重新 tokenize<br/>tokenizer.encode(step["prompt"])<br/>tokenizer.encode(step["response"])
    Note over Trans: ③ overlong_filter: 若 reason in<br/>{TRUNCATION, MAX_STEPS, TIMEOUT}<br/>→ traj_mask 全零化

    rect rgb(220, 235, 255)
        Note over Trans: ④ Prompt 左填充（同 §11.1）
    end

    rect rgb(255, 225, 215)
        Note over Trans: ⑤ Response 右填充（同 §11.1）
    end

    Note over Trans: ⑥ attention_mask / position_ids（同 §11.1）
    Note over Trans: ⑦ loss mask = attention_mask[:, P:] (response 部分)

    rect rgb(220, 250, 220)
        Note over Trans: ⑧ 广播 trajectory_reward 到每步最后 token<br/>score_batch[step_i, resp_len-1] = traj_reward<br/>mc_return_batch[step_i, resp_len-1] = mc_returns[step_i]
    end

    Trans-->>DP: {tensor_batch: {input_ids, attention_mask,<br/>position_ids, responses, prompts,<br/>token_level_scores, mc_returns, response_mask},<br/>non_tensor: {idxs, step_nums, is_last_step,<br/>is_pad_step, step_ids, batch_id},<br/>meta_info: {repeat_counts}}
```

**两路径对比**：

| 维度 | `_transform_agent_trajectories` | `_transform_agent_steps` |
|------|-------------------------------|--------------------------|
| **DataProto 行粒度** | 1 trajectory → 1 行 | 1 step → 1 行 |
| **Tokenization** | 引擎直接提供 token IDs | 重新 `tokenizer.encode()` |
| **MC 回报** | ❌ 无 | ✅ `mc_returns[B, R]` |
| **逐步元数据** | ❌ 无 | ✅ `is_last_step`, `step_ids`, `repeat_counts` |
| **Overlong filter** | ❌（由引擎控制） | ✅ `response_mask` 全零化 |
| **适用算法** | GRPO/REINFORCE（整轨迹） | Stepwise GRPO / Credit Assignment |

---

## 12. Rejection Sampling & Advantage 计算

```mermaid
sequenceDiagram
    participant Batch as DataProto batch
    participant RS as Rejection Sampling
    participant Adv as compute_advantage()
    participant Actor as Actor Update

    Note over Batch: 获得 token_level_scores<br/>（来自环境奖励或 RM 模型）

    rect rgb(255, 235, 220)
        Note over RS: 按 uid 分组（同一任务 n 条 rollout）<br/>检测全对组 (all rewards >= 1): solve_all<br/>检测全错组 (all rewards <= 0): solve_none<br/>记录 partial 组数
    end

    alt rejection_sample.enable = True
        RS->>Batch: 过滤 valid_mask<br/>移除 solve_all + solve_none 的行
        Note over RS: 若过滤后 batch 为空 → skip 该批次
        Note over RS: 向下取整至 world_size 的倍数
    end

    rect rgb(220, 240, 255)
        Note over Batch: Actor forward: compute_log_prob<br/>→ old_log_probs, entropys
        Note over Batch: Ref forward: compute_ref_log_prob<br/>→ ref_log_probs（KL 约束用）
        Note over Batch: token_level_rewards = token_level_scores<br/>（注：KL penalty 以 loss 形式施加，非 reward 扣除）
    end

    alt stepwise_advantage.mode == "broadcast"
        Note over Batch: 分离 last_step 和 other_steps<br/>仅对 last_step 计算优势
        Batch->>Adv: compute_advantage(last_step_batch)
        Note over Adv: GRPO: 按 uid 分组,<br/>A_i = (r_i - mean) / std<br/>REINFORCE: A = r - baseline<br/>RLOO: leave_one_out
        Adv-->>Batch: advantages[last_step]
        Note over Batch: _stepwise_advantage_broadcast:<br/>将 last_step 的 advantage<br/>广播回同 uid 的所有步骤
        Note over Batch: concat(last_step + other_steps)
    else stepwise_advantage.mode == "per_step"
        Note over Batch: uid = step_ids（每步独立分组）<br/>token_level_rewards = mc_returns
        Batch->>Adv: compute_advantage(全步 batch)
        Adv-->>Batch: per-step advantages
    else stepwise_advantage disabled
        Batch->>Adv: compute_advantage(全 traj batch)
        Note over Adv: 标准 GRPO/REINFORCE/RLOO
        Adv-->>Batch: advantages[B]
    end

    Note over Actor: PPO clip update:<br/>ratio = exp(new_log_prob - old_log_prob)<br/>loss = -min(ratio*A, clip(ratio,1±ε)*A)<br/>KL penalty 直接作用于 policy loss
```

**Rejection Sampling 数值示例**（rollout.n=4）：

```
Task q1 的 4 条 rollout 奖励: [1.0, 1.0, 1.0, 1.0]  → solve_all → 丢弃
Task q2 的 4 条 rollout 奖励: [0.0, 0.0, 0.0, 0.0]  → solve_none → 丢弃
Task q3 的 4 条 rollout 奖励: [1.0, 0.0, 1.0, 0.0]  → partial → 保留
Task q4 的 4 条 rollout 奖励: [0.0, 1.0, 0.0, 1.0]  → partial → 保留

metrics:
  batch/solve_none   = 1
  batch/solve_all    = 1
  batch/solve_partial = 2
```

> **动机**：全对组的优势标准差为 0（GRPO 归一化后梯度为 0），全错组同理。保留这些样本只会浪费显存和计算，过滤后可提高单位计算量的信息量。

---

## 13. Stepwise Advantage 训练集成

### 13.1 broadcast 模式（推荐多步任务）

```mermaid
graph TD
    A["每个 Episode (uid=q1)\n3 步, 2 rollouts"]

    A --> B0["rollout 0:\nstep_0, step_1, step_2\ntraj_reward=0.8"]
    A --> B1["rollout 1:\nstep_0, step_1, step_2\ntraj_reward=0.3"]

    B0 --> C0["_transform_agent_steps: 3 行\nis_last_step=[F,F,T]\ntoken_level_scores: 只最后有效"]
    B1 --> C1["_transform_agent_steps: 3 行\nis_last_step=[F,F,T]"]

    C0 & C1 --> D["分离: last_steps=[r0_s2, r1_s2]\nother_steps=[r0_s0, r0_s1, r1_s0, r1_s1]"]

    D --> E["compute_advantage(last_steps)\nuid=q1: A=(0.8-0.55)/σ=+X\nuid=q1: A=(0.3-0.55)/σ=-X"]

    E --> F["_stepwise_advantage_broadcast:\n通过 idxs 将 A 写回同 uid 的所有步"]

    F --> G["concat: 所有步都有 advantages\n→ Actor.update_actor(full_batch)"]
```

### 13.2 per_step 模式（细粒度信用分配）

```mermaid
graph TD
    A["每个 Episode (uid=q1)\n每步独立有 step_reward"]

    A --> B["_transform_agent_steps:\nmc_returns=[γ²·r2, γ·r1+γ²·r2, r1+γ·r1+γ²·r2]"]

    B --> C["token_level_rewards = mc_returns\nuid = step_ids\n(q1_step0, q1_step1, q1_step2)"]

    C --> D["compute_advantage(按 step_id 分组)\nstep_0: A = (mc0_r0 - mc0_r1) / σ\nstep_1: A = (mc1_r0 - mc1_r1) / σ\nstep_2: A = (mc2_r0 - mc2_r1) / σ"]

    D --> E["每步使用自己的 advantage → Actor 更新"]
```

### 13.3 MC Return 计算

Monte Carlo 回报在引擎层（执行轨迹后）计算，写入每步 `Step.mc_return`：

```python
# compute_mc_return(trajectory, gamma)
# 从最后一步反向累积
mc_return = 0.0
for step in reversed(trajectory.steps):
    mc_return = step.reward + gamma * mc_return
    step.mc_return = mc_return
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `gamma` | 折扣因子 | `1.0`（无折扣，适合稀疏奖励） |
| `step.reward` | 单步奖励（通常仅最后步非零） | 由环境 `step()` 返回 |
| `step.mc_return` | 从该步开始的期望累积回报 | 由 `compute_mc_return` 写入 |

---

## 14. 训练阶段完整数据形态

本节汇总每个训练阶段 DataProto 中关键张量的含义与形状（`B`=batch size, `P`=max_prompt_len, `R`=max_response_len）。

```mermaid
graph LR
    subgraph Rollout["推理阶段产出"]
        T1["input_ids [B, P+R]"]
        T2["attention_mask [B, P+R]"]
        T3["position_ids [B, P+R]"]
        T4["prompts [B, P]"]
        T5["responses [B, R]"]
        T6["token_level_scores [B, R]<br/>奖励仅最后有效token非零"]
        T7["response_mask [B, R]<br/>LLM生成token=1, 环境token=0"]
    end

    subgraph Critic["Critic 阶段添加"]
        T8["values [B, R]<br/>价值估计（GAE用）"]
    end

    subgraph LogProb["Log Prob 阶段添加"]
        T9["old_log_probs [B, R]<br/>rollout时的token对数概率"]
        T10["ref_log_probs [B, R]<br/>参考模型的对数概率（KL约束）"]
        T11["token_level_rewards [B, R]<br/>= token_level_scores"]
    end

    subgraph Adv["Advantage 阶段添加"]
        T12["advantages [B, R]<br/>GRPO/REINFORCE/RLOO归一化"]
    end

    subgraph PPO["PPO 更新使用"]
        T13["new_log_probs [B, R]<br/>当前actor的对数概率"]
        T14["ratio = exp(new-old) [B, R]"]
        T15["loss = -min(ratio·A,\nclip(ratio,1±ε)·A)"]
    end

    Rollout --> Critic --> LogProb --> Adv --> PPO
```

**Stepwise 模式额外张量**（仅 `stepwise_advantage.enable=True`）：

| 张量 | 形状 | 说明 |
|------|------|------|
| `mc_returns` | `[B, R]` | Monte Carlo 回报（最后有效token） |
| `is_last_step` | `[B]` (non-tensor) | 该行是否为 episode 的最后一步 |
| `step_ids` | `[B]` (non-tensor) | 步级 uid，格式 `{traj_uid}_step{i}` |
| `is_pad_step` | `[B]` (non-tensor) | 是否为填充 step（对齐 world_size） |
| `idxs` | `[B]` (non-tensor) | 对应原始 batch 的任务索引 |
| `repeat_counts` | `list[int]` (meta) | 每个 episode 展开的步数 |

---

## 15. `hybrid_engine` 配置项详解

### 15.1 含义

`actor_rollout_ref.hybrid_engine` 控制 **Actor（训练）与 Rollout（推理）是否共用同一个 Worker Group**。

```yaml
actor_rollout_ref:
  hybrid_engine: true   # 默认值，rllm 标准模式
```

| 值 | 说明 |
|----|------|
| `true`（混合引擎） | Actor 和 Rollout **合并**在同一个 `actor_rollout_wg` 中，共享 GPU 进程与显存 |
| `false`（分离引擎） | Actor 和 Rollout 分别运行在独立的 `actor_wg` + `rollout_wg` 中 |

### 15.2 各 Trainer 的强制约束

| Trainer | 强制要求 | 原因 |
|---------|---------|------|
| `AgentPPOTrainer` | 必须为 `true` | 依赖异步 Rollout，需混合引擎支持 |
| `AgentWorkflowTrainer` | 必须为 `true` | 同上 |
| `AgentPPOTrainerPipeline` | 必须为 `false` | Pipeline 模式下 Rollout 与 Actor 位于不同 Worker Group |
| `FullyAsyncTrainer` | 必须为 `false` | 全异步架构不支持混合引擎 |

代码位置：

```python
# agent_ppo_trainer.py L54
assert self.config.actor_rollout_ref.hybrid_engine, "Only hybrid engine is supported"

# agent_ppo_trainer_pipeline.py L23
assert not self.hybrid_engine, "PPO pipeline trainer does not support hybrid engine..."
```

### 15.3 对 Batch Padding 的影响

`_pad_dataproto_to_world_size` 根据此标志决定取哪个 Worker Group 的 `world_size` 计算对齐基数：

```python
if self.hybrid_engine:
    world_sizes.append(self.actor_rollout_wg.world_size)   # 合并的 wg
else:
    world_sizes.append(self.actor_wg.world_size)
    world_sizes.append(self.rollout_wg.world_size)
```

### 15.4 工程权衡

| 维度 | `hybrid_engine=true` | `hybrid_engine=false` |
|------|---------------------|----------------------|
| **GPU 利用** | 推理/训练共享显存，无权重传输 | 独立显存，需跨节点同步权重 |
| **延迟** | 低（同进程内切换） | 高（网络传输权重） |
| **规模** | 适合单机多 GPU | 适合超大规模流水线并行 |
| **rllm 支持** | ✅ 默认推荐 | 仅 Pipeline/FullyAsync 模式 |

> **结论**：rllm 的标准训练路径（`AgentPPOTrainer`）强制 `hybrid_engine=true`，即 Actor 训练与 Rollout 推理复用同一组 GPU Worker，无需跨节点传输权重，是 GPU 时间共享机制的基础。
