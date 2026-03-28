# RLLM SDK — 自动 LLM 追踪（Trace）收集与 RL 训练

> 原文件：`rllm/rllm/sdk/README.md`  
> 本文档为面向零基础读者的中文翻译 + 详细解释版本

---

## 📌 先搞懂几个核心概念（看这里！）

在读 README 之前，我们先建立背景知识。这个 SDK 做的事情简单说就是：

> **当你的 AI 程序（Agent）调用大语言模型（LLM）时，自动把每次调用的"输入 + 输出 + 元信息"记录下来，用于后续做强化学习（RL）训练。**

理解下面几个名词就能看懂整个文档：

| 英文术语 | 中文含义 | 类比 |
|---|---|---|
| **Trace** | 追踪记录 | 一次 LLM 调用的"快照"（输入了什么、输出了什么、用了多少 token） |
| **Session** | 会话 | 一个"收集桶"，把一段时间内多次 LLM 调用都归到一起 |
| **Step** | 步骤 | 带有奖励（reward）分数的单次 LLM 调用 |
| **Trajectory** | 轨迹 | 多个 Step 组成一条完整的"解题路径"（用于 RL 训练） |
| **Reward** | 奖励 | 强化学习中的评分，用来教模型"这步做得好/不好" |
| **Decorator** | 装饰器 | Python 的 `@xxx` 语法，给函数加上额外功能 |
| **ContextVar** | 上下文变量 | Python 内置机制，让函数调用链中自动共享"当前状态" |
| **OpenTelemetry** | 分布式追踪标准 | 跨进程、跨服务的追踪协议（业界标准） |

---

## 安装

SDK 是 `rllm` 包的一部分，直接导入即可：

```python
from rllm.sdk import session, get_chat_client, trajectory
```

> 💡 **解释**：不需要额外 `pip install`，装了 `rllm` 就有了。

如果需要 **OpenTelemetry**（分布式追踪）支持，需要安装额外依赖：

```bash
pip install rllm[otel]
```

> 💡 **解释**：`rllm[otel]` 是 Python 的"可选依赖"写法，`otel` 是 OpenTelemetry 的缩写。如果你只在单台机器上跑，不需要装这个。

---

## 快速开始

### 基础用法：Session（会话）

```python
from rllm.sdk import session, get_chat_client

llm = get_chat_client(api_key="sk-...")  # 创建一个被"追踪版"的 LLM 客户端

# 创建一个 session，自动追踪里面所有的 LLM 调用
with session(experiment="v1") as sess:
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    # 查看这个 session 里收集到的所有 trace
    print(f"Collected {len(sess.llm_calls)} traces")
```

> 💡 **解释**：
> - `get_chat_client()` 返回的不是普通的 OpenAI 客户端，而是被"包装"过的版本——每次调用 LLM 时，它会自动把请求和响应记录下来（这就是一条 **Trace**）。
> - `with session(...) as sess:` 是 Python 的上下文管理器语法（`with` 语句）。进入这个块时，SDK 内部会打开一个"收集桶"；退出时关闭。
> - `sess.llm_calls` 就是这个桶里收集到的所有 Trace 列表。
> - `experiment="v1"` 是你自己加的元数据标签，可以任意命名，方便后续筛选数据。

---

### 高级用法：Trajectory 装饰器（轨迹）

```python
from rllm.sdk import trajectory, get_chat_client_async

llm = get_chat_client_async(api_key="sk-...")  # 异步版客户端

@trajectory(name="solver")          # 用 @trajectory 装饰这个函数
async def solve_math_problem(problem: str):
    # 每次 LLM 调用会自动变成轨迹中的一个"步骤"（Step）
    response1 = await llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Solve: {problem}"}]
    )
    response2 = await llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Is this correct?"}]
    )
    return response2.choices[0].message.content

# 注意：加了 @trajectory 后，函数返回值变成了 Trajectory 对象，而不是字符串！
traj = await solve_math_problem("What is 2+2?")
print(f"Steps: {len(traj.steps)}")   # 输出: 2（因为调用了两次 LLM）
traj.steps[0].reward = 1.0           # 给第一步打个奖励分
traj.reward = sum(s.reward for s in traj.steps)  # 整条轨迹的总分
```

> 💡 **解释**：
> - `@trajectory(name="solver")` 是一个**装饰器**，它把 `solve_math_problem` 这个函数"包裹"起来。调用时，SDK 会在内部自动追踪函数运行期间所有的 LLM 调用。
> - 函数原本返回一个字符串，但加了装饰器后，**返回值被替换成了 `Trajectory` 对象**，原始返回值放在 `traj.output` 里。
> - `traj.steps` 是 `Step` 对象的列表，每个 Step 对应一次 LLM 调用。
> - `step.reward = 1.0` 是你在外部给这步打分（相当于告诉 RL 训练器"这步说得对"）。
> - 这个流程就是 **RLHF（基于人类反馈的强化学习）** 的数据收集基础。

---

## 核心概念详解

### 1. Session 后端配置

SDK 支持两种追踪"后端"，通过 `rllm/sdk/config.yaml` 配置：

```yaml
session_backend: "contextvar"   # 或 "opentelemetry"
```

| 后端 | 适用场景 | 原理 |
|---|---|---|
| `contextvar`（默认） | 单进程/单机 | 用 Python 内置的 `contextvars` 模块，在同一进程的调用链中自动传递"当前 session"信息 |
| `opentelemetry` | 多进程/分布式 | 用 W3C 标准的 baggage（行李）机制，通过 HTTP 头把 session 信息传递给其他服务 |

> 💡 **类比**：
> - `contextvar` 就像在同一栋楼里用对讲机——只能楼内通信。  
> - `opentelemetry` 就像通过互联网协议传消息——跨楼甚至跨城市都能同步信息。

---

### 2. Session 上下文（Context）

Session 的作用是**把多次独立的 LLM 调用归到同一个"实验"下**，方便后续调试和分析。

```python
from rllm.sdk import session

with session(experiment="v1") as sess:
    llm.chat.completions.create(...)
    print(sess.llm_calls)   # Trace 对象的列表
```

---

### 3. 元数据继承（Metadata Inheritance）

Session 可以嵌套！内层 session 会自动合并外层的元数据。

```python
with session(experiment="v1"):          # 外层：打标签 experiment="v1"
    with session(task="math"):          # 内层：再加标签 task="math"
        # 这次 LLM 调用的 trace 会同时带有: {experiment: "v1", task: "math"}
        llm.chat.completions.create(...)
```

> 💡 **解释**：类似于给文件打多个标签。外层的标签自动被内层继承合并。在管理大规模实验数据时很有用（实验名 + 任务类型 + 模型版本）。

---

### 4. OpenTelemetry Sessions（分布式追踪）

当你的 Agent 是**分布式架构**（比如：一个"客户端"进程发请求，另一个"服务端"进程调用 LLM），就需要 OpenTelemetry 来跨进程传递 session 信息：

```python
from rllm.sdk.session import otel_session, configure_default_tracer

# 在每个进程启动时配置一次 tracer
configure_default_tracer(service_name="my-agent")

# 客户端：发起 HTTP 请求时，session 信息会自动写入 HTTP 头
with otel_session(name="client") as client_session:
    httpx.post("http://server/api", ...)   # HTTP 头里自动带了 session 信息

# 服务端：从 HTTP 头中自动读取并继承 session 信息
with otel_session(name="handler") as server_session:
    llm.chat.completions.create(...)
    # server_session 自动继承了客户端的 UID 链
```

**OpenTelemetry 的关键特性：**
- 使用 **W3C baggage** 作为 session 状态的唯一来源（W3C 是 Web 标准组织制定的规范）
- HTTP 请求自动携带 context（不需要手动传参）
- 基于 **Span**（跨度）的 UID 体系，与 Jaeger、Grafana Tempo 等主流可观测性工具兼容

> 💡 **Span 是什么**？OpenTelemetry 中最基本的单元，代表"某件事从开始到结束"的时间段。多个 Span 组成一棵树，就是一条完整的分布式追踪链路。

---

### 5. 存储后端

SDK 默认把 trace 存在内存里（程序结束即消失）。可用后端：

| 后端 | 持久化 | 适用场景 |
|---|---|---|
| **InMemorySessionTracer**（默认） | ❌ | 调试、短期任务 |
| **SqliteTracer** | ✅ | 需要持久保存 trace 的场景 |

---

## Proxy 集成（代理层）

SDK 内置了一个**代理模块**，可以把所有 LLM 请求路由到 **LiteLLM Proxy**（一个支持多家 LLM 提供商的统一代理服务器），并在路由时注入元数据。

### 元数据路由中间件

```python
from rllm.sdk.proxy import MetadataRoutingMiddleware

app = MetadataRoutingMiddleware(app)   # 给你的 ASGI 应用包一层中间件
```

> 💡 **解释**：ASGI 是 Python 异步 Web 框架的标准接口（FastAPI、Starlette 等都用它）。这个中间件拦截每个请求，从中提取 session 元数据转发给 LiteLLM。

### LiteLLM 回调钩子

```python
from rllm.sdk.proxy import TracingCallback, SamplingParametersCallback

callbacks = [TracingCallback(), SamplingParametersCallback()]
```

> 💡 **解释**：`TracingCallback` 负责收集 trace，`SamplingParametersCallback` 负责注入采样参数（如 temperature）。

### 元数据 Slug 编码

```python
from rllm.sdk.proxy import encode_metadata_slug, decode_metadata_slug, build_proxied_base_url

metadata = {"session_name": "my-session", "experiment": "v1"}
slug = encode_metadata_slug(metadata)                           # 把字典编码成 URL 安全的字符串
proxied_url = build_proxied_base_url("http://localhost:8000", metadata)  # 构建带元数据的代理 URL
```

> 💡 **解释**：因为 HTTP URL 里不能直接放 JSON 字典，SDK 把元数据编码成 URL 路径的一部分（称为 "slug"），代理服务器收到请求后再解码。

---

## API 速查表

```python
# Session 管理
session(**metadata)            → SessionContext      # 创建一个 session
get_current_session()          → ContextVarSession   # 获取当前 session（仅 contextvar 后端）
get_current_session_name()     → str | None
get_current_metadata()         → dict
get_active_session_uids()      → list[str]           # 获取当前 session 的 UID 链

# OpenTelemetry 专用
otel_session(name=None, **metadata)  → OpenTelemetrySession
configure_default_tracer(service_name="rllm-worker")  → None

# LLM 客户端
get_chat_client(api_key, base_url, ...)       → 同步追踪客户端
get_chat_client_async(api_key, base_url, ...) → 异步追踪客户端

# 装饰器
@trajectory(name: str, **metadata)  # 把函数包装成"轨迹收集器"
```

---

## 数据模型层级

```
单次 LLM 调用
     │
     ▼
  Trace（原始追踪记录）
     中含: trace_id, session_name, input, output, model, tokens, ...
     │
     ▼
   Step（步骤 = Trace + reward 奖励分）
     中含: id, input, output, reward, ...
     │
     ▼
Trajectory（轨迹 = 多个 Step + 总分）
     中含: name, steps[], reward, input（函数参数）, output（函数返回）
```

---

## 运行时助手

```python
from rllm.sdk.session import wrap_with_session_context

# 把 agent 函数包装成"自动附带独立 session context"的版本
wrapped_fn = wrap_with_session_context(agent_func, tracer_service_name="my-agent")
output, session_uid = wrapped_fn(metadata, *args, **kwargs)
```

> 💡 **解释**：在 RL 训练中，`wrap_with_session_context` 确保批量并行运行的每个 agent 调用都有自己独立的 session，不会互相干扰。`session_uid` 用于后续检索这次调用的所有 trace。

---

## 设计原则

| 原则 | 解释 |
|---|---|
| **Minimal API surface** | 常用的就那几个：`session()`、`get_chat_client()`、`@trajectory` |
| **Context-based** | 用 `contextvars` 自动传递当前 session，不需要手动给每个函数传参 |
| **Distributed-ready** | OpenTelemetry 后端支持跨进程追踪 |
| **Pluggable storage** | 支持内存、SQLite，或自定义后端 |
| **Type-safe** | 所有数据模型用 Pydantic 定义，IDE 能自动补全 |
| **Async-native** | 优先支持 `async/await`，适合高并发 Agent 场景 |
| **Proxy-integrated** | 内置 LiteLLM 代理支持，统一管理多家 LLM 提供商 |

---

## 🗺️ 完整数据流

```
你的 Agent 代码
    │  用 @trajectory 装饰函数
    │  用 with session() 创建收集上下文
    ▼
get_chat_client() 返回的"追踪版"LLM 客户端
    │  每次调用 LLM 时自动：
    │  1. 记录 input（发出的 messages）
    │  2. 发请求给 LLM（可经过 LiteLLM Proxy）
    │  3. 记录 output（LLM 回复）和 tokens 用量
    │  4. 生成 Trace，存入当前 Session
    ▼
Session / Trajectory
    │  外部设置 step.reward 和 traj.reward（评分）
    ▼
RL 训练框架（如 verl/PPO）
    │  data_process.py 把 Trajectory 转成训练数据格式
    ▼
        模型参数更新 🎯
```
