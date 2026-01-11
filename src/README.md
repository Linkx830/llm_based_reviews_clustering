# src/ 目录说明

本目录包含所有源代码模块。

## 目录结构

- `app/`: 应用入口和Orchestrator主控
- `agents/`: 各Agent实现（职责单一的处理单元）
- `tools/`: LangChain Tools封装
- `pipelines/`: 流水线定义（执行顺序与依赖）
- `models/`: LLM与Embedding模型封装
- `storage/`: DuckDB读写抽象、表管理、版本字段管理
- `evaluation/`: 自动评估与抽样包生成
- `visualization/`: 图表生成
- `utils/`: 公共工具函数

## 各模块职责

详见各子目录的README.md文件。

