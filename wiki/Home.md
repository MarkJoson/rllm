# rLLM Wiki

rLLM（Reinforcement Learning for Language Models）框架技术文档。

## 入门

| 页面 | 说明 |
|------|------|
| [概览](01-Overview.md) | 框架全貌、核心组件映射、训练管线图 |
| [安装与配置](02-Installation-and-Setup.md) | 系统要求、依赖组、各后端环境 |
| [快速入门](03-Quick-Start-Guide.md) | 三种模式的渐进示例 |

## 核心架构

| 页面 | 说明 |
|------|------|
| [核心架构](04-Core-Architecture.md) | Step→Trajectory→Episode 类型系统、TrajectoryGroup |
| [系统设计与数据流](05-System-Design-and-Data-Flow.md) | 四层架构、GPU 时分复用、数据流详解 |
| [Agent 执行引擎](06-Agent-Execution-Engine.md) | 异步轨迹生成、Token拼装、并行执行 |
| [工作流引擎](07-Workflow-Engine.md) | Workflow 基类、DataProto 转换、池化管理 |
| [SDK 引擎](08-SDK-Engine.md) | LiteLLM Proxy、Trace 收集、框架无关训练 |

## 训练系统

| 页面 | 说明 |
|------|------|
| [Agent 训练器](09-Agent-Trainer.md) | 统一训练入口、后端分派 |
| [Rollout 引擎](10-Rollout-Engines.md) | VerlEngine/OpenAI/Tinker/Fireworks |
| [Chat 模板解析器](11-Chat-Template-Parsers.md) | 自动选择、Thinking Token、tokenize |

## 训练后端

| 页面 | 说明 |
|------|------|
| [VERL 后端](12-VERL-Backend.md) | 分布式PPO/GRPO、GPU共享、FSDP |
| [Tinker 后端](13-Tinker-Backend.md) | 轻量级本地训练 |
| [Fireworks 后端](14-Fireworks-Backend.md) | 远程API训练 |
| [后端对比与选择](15-Backend-Comparison-and-Selection.md) | 决策树、全面对比 |

## Agent 与环境

| 页面 | 说明 |
|------|------|
| [Agent 和环境接口](16-Agent-and-Environment-Interfaces.md) | BaseAgent/BaseEnv 完整API |
| [SWE 域](17-SWE-Domain.md) | Docker化代码修复环境 |
| [Web 导航域](18-Web-Navigation-Domain.md) | 浏览器自动化环境 |
| [数学与代码域](19-Math-and-Code-Domains.md) | 工作流变体、工具Agent |
| [经典 RL 环境](20-Classic-RL-Environments.md) | FrozenLake 示例 |

## 奖励系统

| 页面 | 说明 |
|------|------|
| [奖励函数架构](21-Reward-Function-Architecture.md) | Protocol、RewardOutput、工厂模式 |
| [代码评估](22-Code-Evaluation.md) | 沙盒执行、多数据集评判 |
| [数学评估](23-Math-Evaluation.md) | 评判级联、SymPy符号计算 |
| [系统提示](24-System-Prompts.md) | 域特定提示设计 |

## 数据管理

| 页面 | 说明 |
|------|------|
| [数据集准备](25-Dataset-Preparation.md) | Parquet格式、Dataset类 |
| [数据集类型与格式](26-Dataset-Types-and-Formats.md) | ground_truth格式对照 |
| [轨迹处理](27-Trajectory-Processing.md) | Episode→DataProto转换 |

## 训练配置

| 页面 | 说明 |
|------|------|
| [训练脚本](28-Training-Scripts.md) | 启动方式、关键参数 |
| [配置系统](29-Configuration-System.md) | Hydra/OmegaConf 层次 |
| [Stepwise Advantage](30-Stepwise-Advantage.md) | 逐步优势计算 |

## 高级主题

| 页面 | 说明 |
|------|------|
| [分布式训练与 Ray](31-Distributed-Training-with-Ray.md) | 多GPU/节点配置 |
| [Rejection Sampling](32-Rejection-Sampling-and-Filtering.md) | 样本过滤策略 |
| [蒸馏系统](33-Distillation-System.md) | On-policy 蒸馏 |
| [Trace 收集与调试](34-Trace-Collection-and-Debugging.md) | 日志、监控、排错 |
| [测试基础设施](35-Testing-Infrastructure.md) | pytest 测试用例 |
