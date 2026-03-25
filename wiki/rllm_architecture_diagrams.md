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
