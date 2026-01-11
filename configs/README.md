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

