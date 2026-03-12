# vLLM 超参数调优工具 (vLLM HPT)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org) [![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

本项目是一个面向 **OpenAI 兼容接口 / vLLM 服务** 的超参数调优工具。基于 **AI2 ARC-Challenge** 选择题数据集，通过自动化搜索策略调优 vLLM 模型的采样参数（temperature, top_p, top_k 等），以提升模型在推理和生成任务上的准确率。

项目整合了两种主流调优范式，在统一架构下支持对比和切换：
- **Agent 方法 (A2A)**：采用双代理架构，由 TunerAgent (LLM) 分析错题及历史表现，直接生成下一组参数建议。
- **传统方法 (HPO)**：基于 **Optuna** 优化框架，提供 TPE / GP / CMA-ES / Grid 四种经典的参数空间搜索策略。

---

## 1. 核心特性

- **双引擎调优**：统一 CLI 接口，无缝切换 A2A 模式与传统启发式搜索模式。
- **Thinking 模型支持**：原生支持 `--enable-thinking` 参数，可显式控制服务端的 reasoning 行为。
- **高容错答案解析**：采用多级正则提取策略，兼容 reasoning 模型常见的 Markdown、LaTeX 等非标准输出格式。
- **全周期工作流**：内置中断恢复（Checkpoint）、交互式评估以及基于 WebSocket 的实时 Web 仪表盘。
- **离线可视化分析**：支持通过命令行生成 Optuna 试验历史、参数演化及并行坐标图等可视化图表。

---

## 2. 默认参数约束

系统针对 vLLM / reasoning 模型设定了首轮默认参数。其中 `max_tokens=10240` 是防止 reasoning 模型在思维链阶段耗尽 token 的关键配置。

| 参数 | 默认值 |
|---|---:|
| `temperature` | `0.7` |
| `top_p` | `0.8` |
| `top_k` | `20` |
| `repetition_penalty` | `1.2` |
| `max_tokens` | `10240` |

---

## 3. 环境与依赖

- Python >= 3.10
- 可访问的 OpenAI 兼容推理接口 (vLLM)
- `uv` 包管理器 (推荐)

### 安装配置

```bash
# 安装基础依赖
uv sync

# 安装开发与测试依赖 (含 pytest)
uv sync --extra develop

# 安装 GP 模式所需依赖 (含 torch)
uv sync --extra gp
```

---

## 4. 数据集准备

项目使用 AI2 ARC-Challenge 数据集。预期目录结构如下：

```text
data/ai2_arc/ARC-Challenge/
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

**下载方式 (Python)：**

```python
from datasets import load_dataset
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
dataset["train"].to_json("data/ai2_arc/ARC-Challenge/train.jsonl")
dataset["validation"].to_json("data/ai2_arc/ARC-Challenge/validation.jsonl")
dataset["test"].to_json("data/ai2_arc/ARC-Challenge/test.jsonl")
```

---

## 5. 配置说明

项目通过环境变量进行配置。请复制 `.env.example` 为 `.env` 并更新接口信息：

```bash
cp .env.example .env
```

### 参数说明

```ini
# 答题模型配置（必填，被调优的目标）
EXAM_AGENT_BASE_URL=http://localhost:8000/v1
EXAM_AGENT_API_KEY=EMPTY
EXAM_AGENT_MODEL=your-exam-model

# 调参模型配置（A2A 模式必填，用于提供参数建议）
TUNER_AGENT_BASE_URL=http://localhost:8000/v1
TUNER_AGENT_API_KEY=EMPTY
TUNER_AGENT_MODEL=your-tuner-model
```

---

## 6. 使用方法

### 6.1 运行调优 (`run`)

通过 `--tuning-mode` 参数指定调优策略，支持 `a2a | tpe | gp | cmaes | grid`。

```bash
# 传统模式：TPE 策略，20 轮，每轮 200 题
vllm-hpt run --tuning-mode tpe --rounds 20 --mini-exam-size 200

# Agent 模式：A2A 策略
vllm-hpt run --tuning-mode a2a --rounds 20 --mini-exam-size 200

# 控制推理行为并开启 Dashboard (此参数仅限 a2a 模式)
vllm-hpt run --tuning-mode a2a --enable-thinking --ui --ui-port 8501
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--tuning-mode` | `tpe` | 调优策略：`a2a`, `tpe`, `gp`, `cmaes`, `grid` |
| `--rounds` | `20` | 总调优迭代轮数 |
| `--validation-interval` | `3` | 触发 validation 集评估的轮次间隔 |
| `--concurrency` | `5` | API 请求并发数 |
| `--grid-values` | `3` | grid 策略下每个参数的取值数量 |
| `--wrong-sample-size` | `5` | a2a 模式下提供给 Tuner 的错题样本数 |
| `--enable-thinking` | `None` | 显式开启服务端 reasoning 模式 (extra_body) (仅限 a2a 模式) |

### 6.2 恢复调优 (`resume`)

系统支持从中断的 JSON 检查点恢复调优状态，保留历史记录及随机种子。

```bash
vllm-hpt resume --checkpoint checkpoints/run_20260309_160033.json
```

### 6.3 评估参数 (`evaluate`)

使用指定参数在测试集上执行单次全量评估。

```bash
# 命令行传参
vllm-hpt evaluate --params '{"temperature": 0.7, "top_p": 0.8, "max_tokens": 10240}'

# Web 交互式评估
vllm-hpt evaluate --ui
```

### 6.4 数据可视化 (`visualize`)

从 SQLite/JSON 检查点渲染调优过程分析图表，包括准确率趋势、参数演化及 Optuna 历史图。

```bash
# 使用最新检查点
vllm-hpt visualize

# 指定输出目录
vllm-hpt visualize --checkpoint checkpoints/run_xxx.json --save-dir my_charts/
```

---

## 7. 架构与设计

项目采用高度模块化设计，通过 `TuningRunner` 统一编排数据集抽样、答题并发、答案解析及检查点管理。
关于双代理实现、Optuna 整合逻辑、参数限制策略及三阶段数据集防过拟合设计的详细信息，请参考 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 8. 开发与测试

运行完整的 79 项单元测试（涵盖解析器、参数限制、评估器和状态检查点）：

```bash
uv run pytest tests/ -v
```

**运行时产物说明：**  
运行产生的 `logs/`, `cache/`, `checkpoints/`, `results/` 等目录均已被 `.gitignore` 排除，以防泄露配置或提交噪音数据。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
