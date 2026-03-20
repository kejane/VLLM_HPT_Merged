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

### 安装配置



**1. 安装依赖**

```bash
# 安装基础依赖
uv sync

# 如果需要 GP 算法(含 torch，默认不需要)
uv sync --extra gp
```

**2. 激活虚拟环境**

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (CMD)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

> **提示**：如不执行以上激活，可通过在命令前添加 `uv` 命令运行，例如`uv vllm-hpt run`。

---

## 4. 数据集准备

项目使用 AI2 ARC-Challenge 数据集（已存在于项目中，可跳过）。预期目录结构如下：

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

# Agent 模式：A2A 策略，10 轮，每轮 100 题，并发 10
vllm-hpt run --tuning-mode a2a --rounds 10 --mini-exam-size 100 --concurrency 10

# 控制推理行为并开启 Dashboard
vllm-hpt run --tuning-mode a2a --enable-thinking --ui --ui-port 8501

# Grid Search，每参数 5 个取值，禁用缓存
vllm-hpt run --tuning-mode grid --grid-values 5 --no-cache
```

### 6.2 恢复调优 (`resume`)

系统支持从中断的 JSON 检查点恢复调优状态，保留历史记录及随机种子。可通过参数覆盖原有配置。

```bash
# 从检查点恢复，沿用原配置
vllm-hpt resume --checkpoint checkpoints/run_20260309_160033.json

# 恢复并覆盖部分参数
vllm-hpt resume --checkpoint checkpoints/run_20260309_160033.json --rounds 50 --concurrency 10
```

### 6.3 评估参数 (`evaluate`)

使用指定参数在测试集上执行单次全量评估。

```bash
# 命令行传参
vllm-hpt evaluate --params '{"temperature": 0.7, "top_p": 0.8, "max_tokens": 10240}'

# Web 交互式评估（默认开启 UI）
vllm-hpt evaluate --ui

# 纯命令行评估，不启动 Dashboard
vllm-hpt evaluate --params '{"temperature": 0.7}' --no-ui
```

### 6.4 数据可视化 (`visualize`)

从 SQLite/JSON 检查点渲染调优过程分析图表，包括准确率趋势、参数演化及 Optuna 历史图。

```bash
# 使用最新检查点（自动查找）
vllm-hpt visualize

# 指定检查点与输出目录，仅保存不弹窗
vllm-hpt visualize --checkpoint checkpoints/run_xxx.json --save-dir my_charts/ --no-show
```

---

### 6.5 指令参考总表

#### `run` — 启动新的调优会话

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--tuning-mode` | `str` | `tpe` | 调优策略，可选：`a2a` \| `tpe` \| `gp` \| `cmaes` \| `grid` |
| `--rounds` | `int` | `20` | 总调优迭代轮数 |
| `--mini-exam-size` | `int` | `200` | 每轮 mini-exam 的题目数量 |
| `--validation-interval` | `int` | `3` | 每隔 N 轮触发 validation 集全量评估 |
| `--concurrency` | `int` | `5` | API 请求并发数 |
| `--data-dir` | `str` | `data/ai2_arc/ARC-Challenge` | 数据集目录路径 |
| `--no-cache` | `bool` | `False` | 禁用 API 响应缓存 |
| `--grid-values` | `int` | `3` | Grid Search 模式下每个参数的取值数量 |
| `--seed` | `int` | `None` | 搜索策略的随机种子 |
| `--wrong-sample-size` | `int` | `5` | A2A 模式下每轮提供给 Tuner 的错题样本数 |
| `--output-truncate-length` | `int` | `500` | A2A 模式下每条错题的模型输出最大字符数 |
| `--enable-thinking` / `--disable-thinking` | `bool` | `None` | 显式启用/禁用服务端 reasoning 模式 |
| `--ui` | `bool` | `False` | 启动 WebSocket 实时监控 Dashboard |
| `--ui-port` | `int` | `8501` | Dashboard 服务端口 |

#### `resume` — 从检查点恢复调优

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--checkpoint` | `str` | **必填** | 检查点文件路径 |
| `--tuning-mode` | `str` | `None` | 覆盖调优策略（默认沿用检查点配置） |
| `--rounds` | `int` | `None` | 覆盖总轮数 |
| `--mini-exam-size` | `int` | `None` | 覆盖每轮题目数量 |
| `--validation-interval` | `int` | `None` | 覆盖验证间隔 |
| `--concurrency` | `int` | `None` | 覆盖并发数 |
| `--data-dir` | `str` | `None` | 覆盖数据集目录 |
| `--no-cache` | `bool` | `False` | 禁用 API 响应缓存 |
| `--wrong-sample-size` | `int` | `5` | A2A 模式下错题样本数 |
| `--output-truncate-length` | `int` | `500` | A2A 模式下错题输出最大字符数 |
| `--enable-thinking` / `--disable-thinking` | `bool` | `None` | 启用/禁用 reasoning 模式 |
| `--ui` | `bool` | `False` | 启动 Dashboard |
| `--ui-port` | `int` | `8501` | Dashboard 端口 |

#### `evaluate` — 单次全量测试集评估

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--params` | `str` | `None` | 采样参数 JSON（如 `'{"temperature": 0.7}'`）。省略时配合 `--ui` 从 Dashboard 设置 |
| `--data-dir` | `str` | `data/ai2_arc/ARC-Challenge` | 数据集目录路径 |
| `--concurrency` | `int` | `5` | API 请求并发数 |
| `--enable-thinking` / `--disable-thinking` | `bool` | `None` | 启用/禁用 reasoning 模式 |
| `--ui` / `--no-ui` | `bool` | `True` | 启动/关闭 Dashboard（默认开启） |
| `--ui-port` | `int` | `8501` | Dashboard 端口 |

#### `visualize` — 可视化调优结果

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--checkpoint` | `str` | `None` | 检查点文件路径。省略时自动使用最新检查点 |
| `--save-dir` | `str` | `None` | 图表保存目录（默认 `checkpoints/viz_{run_id}/`） |
| `--no-show` | `bool` | `False` | 仅保存图表，不弹出交互窗口 |

---

## 7. 架构与设计

项目采用高度模块化设计，通过 `TuningRunner` 统一编排数据集抽样、答题并发、答案解析及检查点管理。
关于双代理实现、Optuna 整合逻辑、参数限制策略及三阶段数据集防过拟合设计的详细信息，请参考 [ARCHITECTURE.md](ARCHITECTURE.md)。

---


## 许可证

本项目采用 [MIT 许可证](LICENSE)。
