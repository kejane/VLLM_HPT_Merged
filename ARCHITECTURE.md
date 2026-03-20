# vLLM HPT 架构设计文档

本文档描述 vLLM 超参数调优工具（vLLM HPT）的系统架构、模块职责与核心机制。

> **Code Wins Always**: 本文档内容严格以 `src/vllm_hpt/` 源码为唯一真理（Source of Truth）。

---

## 1. 文档范围与 Source of Truth

本文档涵盖 vLLM HPT 的整体设计，包括编排层、调优层、评估层及模型交互层。所有类名、函数名及数据结构均已通过代码核对。

## 2. 系统总览与分层

系统采用分层架构，旨在整合 Agent 模式 (A2A) 与传统搜索模式 (HPO)。

- **CLI 层 (`cli.py`)**: 处理用户输入、指令分发与环境初始化。
- **编排层 (`orchestrator/`)**: 核心调度逻辑，管理调优循环、数据集加载与状态检查点。
- **调优层 (`tuning/`)**: 提供参数建议策略，支持 `TunerAgent` (A2A) 或基于 Optuna 的 `SearchStrategy` (HPO)。
- **评估层 (`evaluation/`)**: 负责答案解析 (`answer_parser.py`) 与指标计算 (`evaluator.py`)。
- **模型层 (`model/`)**: 封装大模型 API 交互、并发控制与 `enable_thinking` 支持。
- **展现层 (`dashboard/`)**: 提供基于 WebSocket 的实时监控与交互式评估界面。

## 3. src/vllm_hpt/ 文件与文件夹层级

项目代码组织遵循功能模块化原则，核心逻辑位于 `src/vllm_hpt/` 目录下：

### 3.1 核心文件夹职责
- **`orchestrator/`**: 系统的“大脑”。
    - `runner.py`: 定义 `TuningRunner` 类，驱动整个 `sample -> answer -> evaluate -> tune -> checkpoint` 闭环。
    - `checkpoint.py`: `CheckpointManager` 负责 `Checkpoint` 对象的序列化（JSON）与 Optuna 数据库（SQLite）管理。
- **`tuning/`**: 参数搜索逻辑。
    - `params.py`: `SamplingParams` 定义核心调优参数（temperature, top_p, top_k, repetition_penalty, max_tokens）及其 Pydantic 约束。
    - `tuner_agent.py`: A2A 模式核心，将历史记录与错题样本转化为新参数建议。
    - `history.py`: `OptimizationHistory` 存储各轮次的 `HistoryEntry`（参数与准确率对）。
    - `strategies/`: HPO 模式实现，包含 TPE, GP, CMA-ES, Grid 等策略。
- **`model/`**: 通信封装。
    - `client.py`: `LLMClient` 包装 `AsyncOpenAI`，实现指数退避重试与结果缓存。
    - `exam_agent.py`: `ExamAgent` 负责并发调用模型回答 mini-exam。
- **`evaluation/`**: 质量评估。
    - `answer_parser.py`: 提供 `parse_answer` 与 `_detect_parse_level`，支持 5 级正则匹配。
    - `evaluator.py`: 定义 `EvaluationResult`，计算准确率并记录错题细节。
- **`exam/`**: 数据处理。
    - `loader.py`: 加载 ARC-Challenge 等 JSONL 数据集。
    - `sampler.py`: `sample_mini_exam` 实现训练集的随机抽样。
- **`dashboard/`**: UI 后端。
    - `server.py`: 基于 FastAPI/Starlette 的 WebSocket 服务器。
    - `state.py`: 全局状态同步对象，连接 Runner 与前端界面。
- **`utils/`**: 通用工具，包含 `cache.py` (diskcache) 与 `logger.py`。

### 3.2 核心类说明与数据结构
本系统依赖以下核心类进行状态管理与数据传递：
- **`TuningRunner`**: `orchestrator/runner.py`。主控类，持有 `OptimizationHistory` 与 `CheckpointManager`。
- **`SamplingParams`**: `tuning/params.py`。基于 Pydantic，包含 `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens`。
- **`HistoryEntry`**: `tuning/history.py`。单轮实验快照，包含 `round_num`, `params`, `train_accuracy`, `validation_accuracy`。
- **`EvaluationResult`**: `evaluation/evaluator.py`。评估产物，包含 `accuracy`, `correct_count`, `total_count`, `wrong_questions` (List of `WrongQuestion`)。
- **`Checkpoint`**: `orchestrator/checkpoint.py`。持久化数据结构，包含运行 ID、历史记录、最佳参数及随机种子。

## 4. 宏观数据流向

以下展示了 `TuningRunner` 在调优循环中的宏观数据流向：

```text
[ Dataset ] --(load)--> [ Questions ] --(sample)--> [ Mini-Exam ]
                                                         |
                                                         v
[ Suggested Params ] <--(suggest)-- [ TunerAgent / Strategy ]
        |                                                |
        | (params)                                       | (feedback)
        v                                                |
[ ExamAgent ] --(batch call)--> [ LLM Service (vLLM) ]   |
        |                             |                  |
        |                             | (raw outputs)    |
        v                             v                  |
[ AnswerParser ] --(regex extract)--> [ Parsed Answers ] |
                                             |           |
                                             v           |
[ Evaluator ] --(score)--> [ EvaluationResult ] ---------+
                                |
                                +--(add)--> [ OptimizationHistory ]
                                |
                                +--(save)--> [ Checkpoint (JSON) ]
```

1. **抽样**: 从全量训练集中抽取 `mini_exam_size` 题目。
2. **预测**: `ExamAgent` 使用当前 `SamplingParams` 并发请求模型。
3. **解析**: `AnswerParser` 通过多级匹配从长文本（含 Thinking 内容）中提取选项。
4. **评估**: 计算准确率，并产生 `WrongQuestion` 样本。
5. **调优**: 
   - A2A: `TunerAgent` 利用历史记录与错题样本生成 Prompt，请求 Tuner 模型产出新参数。
   - HPO: `SearchStrategy` 将准确率 `report` 给 Optuna，并 `suggest` 下一组参数。
6. **持久化**: `CheckpointManager` 自动保存状态以备中断恢复。

## 5. 核心机制设计

### 5.1 参数钳制与约束 (Pydantic validation)
系统通过 `SamplingParams` (基于 Pydantic) 确保所有参数始终处于物理合法区间。
- **范围限制**: 如 `temperature` (0.0~2.0), `top_p` (0.0~1.0)。
- **自动钳制**: 使用 `@field_validator(mode="before")` 在参数实例化前执行 `clamp` 操作。即使 TunerAgent (LLM) 输出了越界参数（如 `temperature=5.0`），系统也会自动将其修正为 `2.0`。
- **特殊处理**: `top_k=-1` 被定义为禁用该采样限制，在 `to_api_dict` 时会自动从字典中剔除。

### 5.2 缓存与重试机制 (diskcache + backoff/retry)
- **磁盘缓存**: `ResponseCache` 使用 `diskcache` 库。Key 为 `prompt + sorted_params`，Value 为模型输出文本。这极大加快了重复实验的速度并节省了 API 成本。
- **指数退避**: `LLMClient._call_with_retry` 捕获 `openai.APIError`。当服务过载（如 vLLM OOM 或 Rate Limit）时，按 1s, 2s, 4s 等步长自动重试，增强了长时间调优任务的稳定性。

### 5.3 多级正则容错解析
针对 Reasoning 模型（如 DeepSeek-R1）输出内容长、思维链复杂的特点，`AnswerParser` 实现了 5 级回退匹配机制：
1. **Level 1 (Strict)**: 识别 "The best answer is: D" 等标准宣告语句。
2. **Level 2 (Natural)**: 匹配 "answer is **A**" 等带加粗的自然描述。
3. **Level 3 (Boxed)**: 识别 LaTeX 格式的 `\boxed{C}`。
4. **Level 4 (Standalone)**: 匹配行首或行尾的 `A.` 或单独的 `B`。
5. **Level 5 (First Char)**: 极端回退，取去除空格后的首字符。

### 5.4 断点续跑状态持久化
`CheckpointManager` 提供全量状态恢复能力：
- **JSON 状态**: 保存 `run_id`、当前轮次、完整 `OptimizationHistory` 以及当前的最佳参数。
- **随机状态**: 序列化 `random.getstate()`，确保恢复后的 `sample_mini_exam` 抽样序列与中断前一致，保持实验可重复性。
- **HPO 数据库**: 传统搜索模式会自动维护配套的 `.db` (SQLite) 文件，确保存储在 Optuna 中的试验历史不会丢失。

### 5.5 `enable_thinking` 支持
针对 vLLM 的 `enable_thinking` 扩展参数提供原生支持：
- **直通控制**: 在 `SamplingParams` 之外，通过 `LLMClient` 向 API 的 `extra_body` 传递该标志。
- **内容提取**: `_extract_chat_content` 优先从 `choices[0].message.content` 获取内容。如果模型仅在 `reasoning_content` 中输出（如某些预览版 API），系统会自动从中提取文本以供解析。

### 5.6 配置文件与环境集成 (`config.py`)
系统使用 `pydantic-settings` 库进行配置管理，通过 `Settings` 类从 `.env` 文件或环境变量中加载配置。主要包含两类配置：
- **Exam Agent 配置**: 用于答题的基座模型接口（必填），对应环境变量 `EXAM_AGENT_BASE_URL`, `EXAM_AGENT_API_KEY`, `EXAM_AGENT_MODEL`。
- **Tuner Agent 配置**: 仅在 A2A 模式下要求提供，用于参数建议的模型接口。
`get_settings()` 函数采用单例模式缓存配置，确保在整个运行周期内配置的一致性。

### 5.7 结果导出与分析 (`visualization.py`)
除了实时 Dashboard，系统还支持通过 `vllm-hpt visualize` 命令行工具进行离线深度分析：
- **Optuna 历史图**: 展示搜索随时间的变化趋势。
- **参数重要性**: 分析哪些采样参数对准确率影响最大。
- **平行坐标图**: 揭示不同参数取值组合与最终得分的关联。

---

## 6. 微观算法流程

### 6.1 主调优循环 (Main Tuning Loop)

`TuningRunner._run_loop` 是系统的核心驱动引擎，负责协调各模块完成一轮完整的参数评估与迭代。

```text
[ Start Round ]
       |
       v
sample_mini_exam(train_questions) -> List[Question]
       |
       v
ExamAgent.answer_questions(mini_exam, current_params) -> List[str] (raw_outputs)
       |
       v
evaluate(mini_exam, raw_outputs) -> EvaluationResult
       |
       v
HistoryEntry(round, params, accuracy) -> OptimizationHistory.add_entry()
       |
       v
(If interval) _run_validation() -> Update Best Params
       |
       v
suggest_params / suggest_next_params -> New SamplingParams
       |
       v
CheckpointManager.save() -> [ JSON Checkpoint ]
       |
       v
[ Next Round ]
```

**执行流详情**:
`sample_mini_exam` (从全量训练集中随机抽取题目) -> `List[Question]` (本次评估的题目子集) -> `ExamAgent.answer_questions` (并发调用模型获取原始回答) -> `raw_outputs` (模型输出字符串列表) -> `evaluate` (解析并计算准确率) -> `EvaluationResult` (包含准确率、错题、解析统计的对象) -> `HistoryEntry` (封装当前轮次的执行快照) -> `OptimizationHistory` (更新全局历史记录) -> `_run_loop` (根据 `tuning_mode` 调度建议引擎获取 `SamplingParams`) -> `CheckpointManager.save` (保存当前进度)。

---

### 6.2 A2A 建议流 (Agent-to-Agent Suggestion)

`TunerAgent.suggest_params` 利用大模型的推理能力，通过分析过去的失败案例来优化未来参数。

```text
[ OptimizationHistory ] + [ EvaluationResult (Wrong Questions) ]
       |
       v
TunerAgent._build_prompt() -> str (Full Context Prompt)
       |
       v
LLMClient.chat(messages) -> str (LLM JSON Response)
       |
       v
TunerAgent._parse_response() -> SamplingParams (Clamped)
       |
       v
(Fallback if Error) random_perturbation(current_params)
```

**执行流详情**:
`suggest_params` (接收历史记录与当前评估结果) -> `_build_prompt` (将参数范围说明、历史表现、Top 5 轮次、5 个典型错题样本拼接成 Prompt) -> `str(Prompt)` (提交给调参模型) -> `LLMClient.chat` (执行 API 调用) -> `str(LLM JSON Response)` (原始 JSON 字符串) -> `_parse_response` (执行正则提取与 `json.loads`) -> `SamplingParams` (经过 Pydantic 自动钳制后的新参数对象)。若连续 3 次解析失败，系统调用 `random_perturbation` 进行随机微调作为回退。

---

### 6.3 答案解析流 (Multi-level Answer Parsing)

`AnswerParser` 负责从复杂的模型输出（包含思维链、LaTeX、自然语言）中精准提取 [A-D] 标签。

```text
[ raw_output (String) ]
       |
       v
_detect_parse_level(Level 1: Strict Regex) -> Optional[str]
       | (if None)
       v
_detect_parse_level(Level 2: Natural Phrases) -> Optional[str]
       | (if None)
       v
_detect_parse_level(Level 3: Boxed \boxed{}) -> Optional[str]
       | (if None)
       v
_detect_parse_level(Level 4: Standalone A. / B) -> Optional[str]
       | (if None)
       v
_detect_parse_level(Level 5: First Non-space Char) -> Optional[str]
```

**执行流详情**:
`raw_outputs` (待解析字符串列表) -> `parse_answers` (循环处理) -> `_detect_parse_level` (逐级匹配) -> `Optional[str]` (解析出的 A/B/C/D 或 None)。系统会记录每条回答命中的 Level（1-5），并在 `AnswerParserStats` 中统计各级成功率。对于 Reasoning 模型，Level 3 (`\boxed{}`) 和 Level 1 (`The best answer is:`) 是最常见的命中点。

---

### 6.4 状态持久化流 (State Persistence)

`CheckpointManager` 确保调优过程可以在崩溃或手动中断后完美恢复。

```text
[ TuningRunner State ]
       |
       v
Checkpoint(run_id, round, history, best_params, random_state)
       |
       v
CheckpointManager.save() -> json.dump(indent=2)
       |
       v
[ file_run_TIMESTAMP.json ]
       |
       ^
CheckpointManager.load() -> Checkpoint Dataclass
       |
       v
random.setstate(checkpoint.random_state) -> resume _run_loop
```

**执行流详情**:
`TuningRunner` 在每轮结束时调用 `_save_checkpoint` -> `CheckpointManager.save` (构建 `Checkpoint` 实例) -> `Checkpoint` Dataclass (包含运行 ID、轮数、`OptimizationHistory.to_dict()`、`best_params.model_dump()` 及 `random.getstate()`) -> `JSON file` (磁盘持久化)。恢复时，`CheckpointManager.load` 将 JSON 反序列化并重建对象，Runner 显式恢复 `random` 模块的状态，确保抽样一致性。

---

### 6.5 HPO Ask-Tell 流 (Optuna Search Flow)

`SearchStrategy` 封装了 Optuna 的“询问-告知”模式，支持 TPE, GP 等多种传统优化算法。

```text
[ Create Strategy ] -> optuna.create_study(direction="maximize")
       |
       v
SearchStrategy.suggest_next_params() -> optuna.trial.Trial
       |
       v
[ Run Evaluation ] -> accuracy (float)
       |
       v
SearchStrategy.report_result(params, accuracy) -> optuna.study.tell()
       |
       v
[ Optuna SQLite DB Update ]
```

**执行流详情**:
`TuningRunner` 初始化对应的子类（如 `TPESearchStrategy`）-> `suggest_next_params` (通过 Optuna 的 `ask` API 获取参数建议) -> `optuna.trial.Trial` (内部生成的试验对象) -> `SamplingParams` (基于搜索空间定义的建议值) -> `evaluate` (获取模型表现) -> `report_result` (将评估结果反馈给策略) -> `optuna.study.tell` (告知 Optuna 该试验的得分) -> `OptimizationHistory` (同步记录以便 A2A 模式切换时仍能看到历史)。

---

## 7. Dashboard / Visualization

系统通过 `vllm_hpt.dashboard` 提供 Web 端监控能力。
- **实时同步**: `TuningRunner` 通过 `get_dashboard_state()` 修改全局单例，前端通过 WebSocket 订阅更新。
- **健康监控**: 展示实时准确率曲线、当前采样参数、调优进度以及 Exam/Tuner 模型的健康状态。
- **交互评估**: 在 `evaluate` 模式下，支持用户通过 Web 界面手动输入参数并触发测试集全量评估。
- **离线绘图**: 调优结束后，可通过 `vllm-hpt visualize` 生成 Optuna 历史图、参数平行坐标图及准确率分布图。

## 8. 附录：关键术语表

| 术语 | 定义 | 关联代码 |
| :--- | :--- | :--- |
| **A2A** | Agent-to-Agent，指使用 LLM (Tuner) 调优另一个 LLM (Exam) 的模式。 | `TunerAgent` |
| **HPO** | Hyperparameter Optimization，指使用传统算法（如 TPE）搜索最优参数。 | `SearchStrategy` |
| **Mini-Exam** | 每一轮调优中从训练集随机抽取的题目子集，用于快速评估参数质量。 | `sample_mini_exam` |
| **Source of Truth** | 本项目坚持以 `src/vllm_hpt/` 下的 Python 源码作为架构设计的唯一真理。 | `vllm_hpt/` |
| **Thinking Support** | 原生支持推理型模型（如 DeepSeek-R1）的 `reasoning_content` 提取。 | `LLMClient.enable_thinking` |
| **Checkpoint** | 包含完整历史记录、最佳参数及随机种子状态的序列化快照。 | `CheckpointManager` |

---
(本文档更新于 2026-03-20)
