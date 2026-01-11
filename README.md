# 基于电商评论的方面级情感分析、需求聚类与多Agent自动洞察系统

## 项目简介

本项目是一个基于LangChain + DuckDB + LLM的数据挖掘系统，用于从海量电商评论中抽取方面级情感、聚类痛点需求，并自动生成可执行的洞察建议。

## 数据前提

- DuckDB数据库文件：`data/duckDB/amazon.duckdb`
- 数据表：
  - `reviews`：用户评论表（评分、标题、正文、时间等）
  - `meta`：商品元数据表（类目、标题、features等）
- 连接键：`reviews.parent_asin = meta.parent_asin`

## 快速开始

### 1. 环境准备

#### 安装依赖
```bash
# 使用uv安装依赖
uv sync

# 或使用pip
pip install -e .
```

#### 配置环境变量
复制`.env.example`为`.env`并填写必要的配置：
```bash
cp .env.example .env
```

编辑`.env`文件，至少需要配置：
- **使用 Ollama（本地部署）**：
  - `LLM_PROVIDER=ollama`
  - `LLM_MODEL=qwen2.5:4b`（或您部署的其他模型名称）
  - `LLM_BASE_URL=http://localhost:11434`（可选，默认值）
- **使用 OpenAI**：
  - `LLM_PROVIDER=openai`
  - `LLM_MODEL=gpt-3.5-turbo`
  - `API_KEY=your_api_key_here`
- `DUCKDB_PATH`: DuckDB数据库文件路径（默认：`data/duckDB/amazon.duckdb`）

### 2. 准备运行配置

创建或修改运行配置文件（`configs/runs/`），例如：
```yaml
# configs/runs/my_run.yaml
data_slice:
  main_category: "Appliances"
  limit: 1000

models:
  # 为不同任务配置不同的模型
  extraction_llm:
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.0
  insight_llm:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.3
  embedding_model: "all-MiniLM-L6-v2"

prompts:
  extraction: "v1.0"
  insight: "v1.0"

clustering:
  method: "kmeans"
  n_clusters: 10
```

### 3. 运行流水线

```bash
# 基本运行
python main.py --config configs/runs/example.yaml

# 指定运行ID
python main.py --config configs/runs/example.yaml --run-id my_run_001

# 从指定步骤恢复（断点续跑）
python main.py --config configs/runs/example.yaml --resume-from extraction

# 从clustering步骤恢复运行（使用已有的run_id）
# 步骤1：先清理clustering及之后的数据（避免重复）
python scripts/cleanup_resume.py --run-id <your_run_id> --step clustering
# 步骤2：使用原run_id从clustering步骤恢复运行
python main.py --config configs/runs/example.yaml --run-id <your_run_id> --resume-from clustering

# 设置日志级别
python main.py --config configs/runs/example.yaml --log-level DEBUG
```

### 4. 查看结果

运行完成后，结果将保存在：
- `outputs/runs/<run_id>/final_report.md`: 最终报告
- `outputs/runs/<run_id>/evaluation/evaluation_metrics.csv`: 评估指标
- DuckDB中间表：可通过SQL查询查看各步骤的中间结果

## 输出说明

### DuckDB中间表
- `selected_reviews`：实验样本固定表
- `review_sentences`：句子表（含上下文窗口）
- `aspect_sentiment_valid`：校验后的方面情感事实表
- `issue_clusters`：聚类归属表
- `cluster_reports`：簇洞察表
- 更多表见`docs/附录 B：DuckDB 中间表与版本字段规范.md`

### 输出文件
- `outputs/runs/<run_id>/final_report.md`：最终报告
- `outputs/runs/<run_id>/evaluation/evaluation_metrics.csv`：评估指标
- `outputs/runs/<run_id>/manifest.json`：产物清单

## 复现说明

### 指定run_id
```python
from src.app import Orchestrator

orchestrator = Orchestrator(
    db_path="data/duckDB/amazon.duckdb",
    run_id="20260103-1530_category_headphones_v11"
)
```

### 指定数据切片
在运行配置文件中设置：
- `data_slice.main_category`：主类目
- `data_slice.parent_asin`：商品父ID
- `data_slice.time_window`：时间窗口
- `data_slice.filters`：过滤条件

### 指定模型与Prompt版本
在运行配置文件中设置：

#### 为不同任务配置不同的模型（推荐）
系统支持为不同任务配置不同的LLM模型，以满足不同任务的需求：

```yaml
models:
  # 抽取任务模型（需要精确的结构化输出）
  extraction_llm:
    provider: "ollama"
    model: "qwen3:8b"  # 推荐使用较小但精确的模型
    base_url: "http://localhost:11434"
    temperature: 0.0  # 低温度，更精确
    enable_reasoning: false  # 是否启用reasoning模式（默认false，启用会增加推理时间但可能提高质量）
  
  # 洞察任务模型（需要生成性文本）
  insight_llm:
    provider: "ollama"
    model: "qwen3:14b"  # 推荐使用更大的模型，更有创造性
    base_url: "http://localhost:11434"
    temperature: 0.3  # 稍高温度，更有创造性
    enable_reasoning: false  # 是否启用reasoning模式（默认false，启用会增加推理时间但可能提高质量）
```

**任务说明**：
- **抽取任务**（`AspectSentimentExtractorAgent`）：需要精确的结构化输出，推荐使用较小但精确的模型，temperature=0.0
- **洞察任务**（`ClusterInsightAgent`）：需要生成性文本，推荐使用更大的模型，temperature=0.3

**Reasoning模式**：
- `enable_reasoning`：是否启用reasoning模式（默认false）
  - 启用后，模型会进行更深入的推理，可能提高输出质量，但会增加推理时间和成本
  - 适用于需要高质量输出的场景，如复杂的抽取任务或洞察生成
  - 每个LLM配置都可以独立设置此选项

#### 使用统一模型（向后兼容）
如果未指定`extraction_llm`或`insight_llm`，系统会使用以下默认配置：

```yaml
models:
  llm_provider: "ollama"
  llm_model: "qwen3:8b"
  llm_base_url: "http://localhost:11434"
  enable_reasoning: false  # 默认LLM的reasoning模式（默认false）
```

#### Prompt版本配置
- `prompts.extraction`：抽取Prompt版本
- `prompts.insight`：洞察Prompt版本

## 项目结构

```
project/
├── configs/          # 配置文件
│   ├── runs/        # 运行配置
│   ├── prompts/     # Prompt模板
│   ├── schemas/     # Schema定义
│   ├── taxonomy/    # 词表
│   └── clustering/  # 聚类配置
├── src/             # 源代码
│   ├── app/         # 主控（Orchestrator）
│   ├── agents/      # 各Agent实现
│   ├── tools/       # LangChain工具
│   ├── pipelines/   # 流水线定义
│   ├── models/      # LLM/Embedding封装
│   ├── storage/     # DuckDB管理
│   └── utils/       # 工具函数
├── outputs/         # 输出目录
│   └── runs/        # 每次运行的产出
├── docs/            # 文档
└── data/            # 数据目录
```

## 依赖安装

### 使用uv（推荐）
```bash
uv sync
```

### 使用pip
```bash
pip install -e .
```

### 主要依赖
- `langchain>=1.2.0`: LangChain框架
- `duckdb>=1.4.3`: DuckDB数据库
- `sentence-transformers>=5.2.0`: Embedding模型
- `scikit-learn>=1.7.2`: 机器学习工具（聚类）
- `hdbscan>=0.8.41`: HDBSCAN聚类算法
- `pydantic>=2.12.5`: 数据验证
- `pyyaml>=6.0.3`: YAML配置文件解析

## 详细文档

- **设计文档**：`docs/软件设计文档.md` - 系统架构和设计说明
- **表规范**：`docs/附录 B：DuckDB 中间表与版本字段规范.md` - 数据库表结构
- **Schema规范**：`docs/附录 A：LLM 结构化输出 Schema 规范.md` - LLM输出格式
- **文件规范**：`docs/项目文件规范.md` - 项目文件组织规范

## 常见问题

### Q: 如何查看运行日志？
A: 日志会输出到控制台，同时保存在`outputs/runs/<run_id>/logs/`目录。

### Q: 如何复现之前的运行结果？
A: 使用相同的`run_id`和配置文件即可复现。系统会检查中间表，跳过已完成的步骤。

### Q: LLM调用失败怎么办？
A: 系统内置了重试机制（默认3次，指数退避）。如果持续失败，请检查：
- API密钥是否正确
- API配额是否充足
- 网络连接是否正常

### Q: 如何调整聚类参数？
A: 在运行配置文件的`clustering`部分修改参数，例如：
```yaml
clustering:
  method: "hdbscan"  # 或 "kmeans"
  min_cluster_size: 5
  n_clusters: 10
```

### Q: 如何添加自定义的同义词或噪声词？
A: 编辑`configs/taxonomy/`目录下的CSV文件：
- `aspect_synonyms.csv`: 同义词映射
- `noise_terms.csv`: 噪声词列表
- `aspect_allowlist.csv`: 允许的aspect列表

## 版本信息

- Pipeline Version: v1.0
- 最后更新: 2026-01-03

