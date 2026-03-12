# vLLM HPT 架构设计文档

本文档描述 vLLM 超参数调优工具的系统架构、模块职责、数据流向以及 A2A 模式与传统 HPO 模式的整合设计，旨在为后续的维护与扩展提供参考依据。

---

## 1. 系统概述

本项目实现了一个高度自动化的超参数调优流水线，用于优化大语言模型（LLM）的采样参数（temperature, top_p, top_k 等）。系统以 **AI2 ARC-Challenge** 数据集作为反馈基准，计算准确率作为目标函数。

核心架构将两条调优路线整合在同一套评测基建之上：
1. **Agent 模式 (A2A)**：基于 Tuner Agent (LLM) 分析历史记录与错误模式，生成参数建议。
2. **传统搜索模式 (HPO)**：基于 Optuna 框架，利用数学优化策略 (TPE, GP, CMA-ES, Grid) 搜索最优解。

两类模式共享数据集加载、并发答题、多级解析、检查点管理和可视化评估流程。

---

## 2. 模块架构与分层设计

系统采用严格的分层架构，解耦调度逻辑与业务实现：

```text
┌─────────────────────────────────────────────────────────────────┐
│                           CLI 层                                │
│                          (cli.py)                               │
├─────────────────────────────────────────────────────────────────┤
│                          编排层                                 │
│               (runner.py, checkpoint.py)                        │
├─────────────────────────────────────────────────────────────────┤
│                          代理层                                 │
│             (exam_agent.py, tuner_agent.py)                     │
├─────────────────────────────────────────────────────────────────┤
│                          核心逻辑层                             │
│     (loader.py, sampler.py, evaluator.py, answer_parser.py,     │
│      history.py, params.py, strategies/)                        │
├─────────────────────────────────────────────────────────────────┤
│                         基础设施层                              │
│  (config.py, client.py, cache.py, logger.py, dashboard/, viz)   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 模块职责定义

| 模块名称 | 文件路径 | 核心职责 |
|---|---|---|
| **命令行接口** | `cli.py` | 暴露终端指令集（run, resume, evaluate, visualize），统一参数解析。 |
| **主控编排** | `orchestrator/runner.py` | 驱动调优循环，管理验证周期，调度策略分派与状态序列化。 |
| **状态持久化** | `orchestrator/checkpoint.py` | 管理运行状态的序列化与反序列化，保障调优断点续跑能力。 |
| **模型通信** | `model/client.py` | 封装 OpenAI 兼容接口，提供指数退避重试、响应缓存及特定字段（如 `enable_thinking`）注入。 |
| **答题代理** | `model/exam_agent.py` | 依据题目构造 prompt 模板，利用信号量实现并发答题控制。 |
| **调参代理** | `tuning/tuner_agent.py` | 负责 A2A 模式的参数建议，构建包含错题与历史趋势的上下文 prompt。 |
| **传统搜索策略** | `tuning/strategies/*.py` | 封装 Optuna 的 `ask-and-tell` API，实现数学搜索逻辑。 |
| **参数约束** | `tuning/params.py` | 定义参数模型（Pydantic），负责边界检查与非法值归一化 (clamp)。 |
| **答案提取** | `evaluation/answer_parser.py` | 实施多级正则匹配，处理 LLM 的复杂输出（如 Markdown, LaTeX 格式）。 |
| **成效评估** | `evaluation/evaluator.py` | 执行标答比对，生成准确率指标及详细错题报告。 |
| **实时监控** | `dashboard/*` | 提供基于 FastAPI 和 WebSocket 的状态监控界面。 |

---

## 3. 统一调度流程

无论是 A2A 模式还是传统模式，均遵循相同的调优循环（Tuning Loop）：

```text
[数据准备] sample_mini_exam(train, n=200)
    ↓
[模型推理] ExamAgent.answer_questions(params)
    ↓
[指标评估] evaluate(raw_outputs) -> EvaluationResult (含错题详情)
    ↓
[状态记录] HistoryEntry 写入
    ↓
[策略分流] 
    ├─ A2A: TunerAgent.suggest_params(history, wrong_questions)
    └─ HPO: Strategy.report_result() -> suggest_next_params()
    ↓
[参数更新] 生成新的 SamplingParams
    ↓
[状态保存] CheckpointManager.save()
```

### 3.1 验证阶段 (Validation Interval)
系统按照设定的频率（如 `validation_interval=3`）触发全量验证集的测试。该阶段的准确率用于确定当前周期内的最佳参数组合，防止在训练集上发生过拟合。

---

## 4. 关键机制设计

### 4.1 参数系统与合法性边界
系统对采样参数实施严格的自动钳制（Auto-clamp）策略。当 TunerAgent 生成的 JSON 或 Optuna 采样的值超出预设边界时，系统将自动纠正。

- **边界范围**：`temperature` [0.0, 2.0], `top_p` [0.0, 1.0], `top_k` [-1, 100]
- **大上下文要求**：为适配 reasoning 模型特质，`max_tokens` 默认并允许上限调整至 `10240`，以防止输出截断导致答案解析失败。

### 4.2 鲁棒的答案解析器
考虑到 reasoning 风格模型即使在严格指令下也容易产生带有解释步骤或标记的输出，`answer_parser.py` 实施了 5 级递进解析逻辑：
1. 严格模式：`The best answer is: D`
2. 自然语言：`answer is **D**` / `Choice B`
3. 数学/LaTeX 格式：`\boxed{C}`
4. 后缀句号：`A.`
5. 首字母提取

### 4.3 状态恢复的正确性保证
从 checkpoint 恢复不仅恢复先前的最佳参数与历史数据，还会显式恢复 Python 的 `random.setstate`，以确保数据抽样的可复现性。A2A 模式恢复时会自动读取历史中最近一次的参数配置，而 HPO 模式则从 Optuna SQLite 中恢复。

### 4.4 `enable_thinking` 显式控制
`model/client.py` 针对特定服务端模型增加了 `extra_body["enable_thinking"]` 的下发支持。这允许用户通过命令行直观控制模型的推理链路开关。对空返回（`content=None`）的情况，客户端具备自动拉取 `reasoning_content` 的容错机制。

---

## 5. 扩展性设计

项目在设计上预留了标准化的扩展接口：

- **接入新策略**：在 `tuning/strategies/` 下实现 `SearchStrategy` 接口，并在 CLI 侧完成参数透传即可。
- **拓展参数空间**：通过扩展 `tuning/params.py` 内的 `SamplingParams` 及相应的 `to_api_dict`，即可使所有调参模式感知新参数。
- **多数据集支持**：`exam/loader.py` 定义了统一的 `Question` 数据类，只需实现相应的 JSONL 读取解析逻辑即可接入其余测评任务。

---

## 6. 运行产物与隔离策略

为避免状态污染和敏感数据泄漏，系统的所有的动态输出均未受 Git 追踪管理。涵盖路径包括：
- `.env`（模型 API 凭证）
- `checkpoints/`（运行断点及 SQLite DB）
- `logs/`（执行轨迹）
- `results/`（评估报告）
- `cache/`（响应缓存 DB）
