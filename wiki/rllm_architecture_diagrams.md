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
