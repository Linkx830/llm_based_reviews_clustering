# configs/ 目录说明

本目录包含所有配置文件。

## 目录结构

- `runs/`: 每次实验运行配置（推荐一run一个yaml）
- `prompts/`: Prompt模板（结构化抽取/洞察/复核等）
  - `extraction/`: 抽取Prompt
  - `insight/`: 洞察Prompt
  - `judge_recheck/`: 复核Prompt
  - `report/`: 报告Prompt
- `schemas/`: 结构化输出Schema（JSON Schema）
- `taxonomy/`: aspect同义词表、噪声词表等
- `clustering/`: 聚类参数配置（KMeans/HDBSCAN等）

## 配置规范

详见`docs/项目文件规范.md`第3节。

## 配置文件类型

### 1. LLM方法配置（主线）

使用 `example.yaml` 作为模板，配置LLM模型、prompt版本等。

### 2. 传统方法配置（Baseline）

使用 `traditional_baseline.yaml` 作为模板，配置传统NLP抽取方法。

**关键配置项：**
- `run.use_traditional: true` - 启用传统baseline模式
- `run.pipeline_version: "traditional_v1.0"` - 传统方法pipeline版本
- `traditional.extraction.method: "LEXICON_RULE"` - 抽取方法
- `traditional.extraction.aspect_seed_lexicon` - aspect种子词表（可选）
- `traditional.insight.method_version: "v1.0"` - 洞察方法版本

**注意事项：**
- 传统方法不需要配置LLM模型（`extraction_llm`、`insight_llm`）
- 但必须配置embedding模型（用于聚类，需与LLM方法保持一致以保证公平对比）
- 聚类配置应与LLM方法保持一致（`clustering`部分）
- 建议与LLM方法使用相同的数据切片（`data_slice`）以保证公平对比

## 运行方式

```bash
# LLM方法
python main.py --config configs/runs/example.yaml

# 传统方法
python main.py --config configs/runs/traditional_baseline.yaml
```

