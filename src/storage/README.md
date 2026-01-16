# storage/ 模块说明

## 职责

DuckDB存储管理模块，负责：
- DuckDB连接管理（单写者模式）
- 表存在性检查、表结构定义
- 版本字段生成与管理
- run_id过滤视图

## 文件说明

- `connection.py`: DuckDBConnection - 连接管理器（单例模式）
  - `execute_read()`: 执行只读查询，支持位置参数列表（DuckDB使用?占位符）
  - `execute_write()`: 执行写入操作，串行化保证单写者模式
- `table_manager.py`: TableManager - 表管理（表名常量、表结构定义）
- `version_fields.py`: VersionFields - 版本字段生成（run_id、data_slice_id等）

## 注意事项

- DuckDB使用`?`占位符和位置参数（列表），不支持`:parameter`命名参数语法
- `execute_read()`和`execute_write()`都接受列表参数，字典参数会自动转换为列表

## 使用示例

```python
from src.storage import DuckDBConnection, TableManager

# 获取连接
db = DuckDBConnection.get_instance("data/duckDB/amazon.duckdb")
db.connect()

# 创建表
table_manager = TableManager(db)
table_manager.create_all_tables()
```

## 数据表结构说明

### 主线表（LLM方法）

主线表用于LLM方法的流水线，包括以下表：

### 1. selected_reviews（选中的评论表）

存储从数据源筛选出的原始评论数据。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID，标识本次数据处理的唯一标识 |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| source_snapshot_at | TIMESTAMP | 否 | 数据源快照时间 |
| review_pk | VARCHAR | 是 | 评论主键（唯一标识） |
| parent_asin | VARCHAR | 是 | 父产品ASIN |
| asin | VARCHAR | 否 | 子产品ASIN（变体产品） |
| user_id | VARCHAR | 否 | 用户ID |
| timestamp | INTEGER | 是 | 评论时间戳（Unix时间戳） |
| rating | DOUBLE | 是 | 评分（1-5星） |
| verified_purchase | BOOLEAN | 否 | 是否已验证购买 |
| helpful_vote | INTEGER | 否 | 有用票数 |
| review_title | VARCHAR | 否 | 评论标题 |
| review_text | VARCHAR | 是 | 评论正文 |
| main_category | VARCHAR | 否 | 产品主分类 |
| product_title | VARCHAR | 否 | 产品标题 |

### 2. meta_context（元数据上下文表）

存储产品的元数据信息，用于提供上下文。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| parent_asin | VARCHAR | 是 | 父产品ASIN |
| product_title | VARCHAR | 是 | 产品标题 |
| main_category | VARCHAR | 否 | 产品主分类 |
| features_short | VARCHAR | 否 | 产品特性摘要 |
| description_short | VARCHAR | 否 | 产品描述摘要 |
| details_short | VARCHAR | 否 | 产品详情摘要 |
| context_version | VARCHAR | 是 | 上下文版本号 |

### 3. normalized_reviews（规范化评论表）

存储经过文本清理和规范化处理的评论数据。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| review_pk | VARCHAR | 是 | 评论主键 |
| parent_asin | VARCHAR | 是 | 父产品ASIN |
| timestamp | INTEGER | 是 | 评论时间戳 |
| rating | DOUBLE | 是 | 评分 |
| clean_text | VARCHAR | 是 | 清理后的评论文本 |
| cleaning_flags | VARCHAR | 否 | 清理标记（记录清理操作类型） |

### 4. review_sentences（评论句子表）

存储将评论拆分后的句子数据，包含上下文信息。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| sentence_id | VARCHAR | 是 | 句子ID（唯一标识） |
| review_pk | VARCHAR | 是 | 所属评论主键 |
| parent_asin | VARCHAR | 是 | 父产品ASIN |
| timestamp | INTEGER | 是 | 评论时间戳 |
| rating | DOUBLE | 是 | 评分 |
| verified_purchase | BOOLEAN | 否 | 是否已验证购买 |
| helpful_vote | INTEGER | 否 | 有用票数 |
| sentence_index | INTEGER | 是 | 句子在评论中的索引位置 |
| target_sentence | VARCHAR | 是 | 目标句子文本 |
| prev_sentence | VARCHAR | 否 | 前一句子文本（上下文） |
| next_sentence | VARCHAR | 否 | 后一句子文本（上下文） |
| context_text | VARCHAR | 是 | 完整上下文文本 |

### 5. opinion_candidates（观点候选表）

存储筛选出的观点候选句子，用于后续提取。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| sentence_id | VARCHAR | 是 | 句子ID |
| is_candidate | BOOLEAN | 是 | 是否为候选（True/False） |
| filter_reason | VARCHAR | 否 | 过滤原因（如果被过滤） |
| priority_weight | DOUBLE | 否 | 优先级权重 |

### 6. aspect_sentiment_raw（原始aspect和情感提取表）

存储LLM原始提取的aspect和情感信息，包含解析状态和错误信息。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| llm_model | VARCHAR | 是 | 使用的LLM模型名称 |
| prompt_version | VARCHAR | 是 | 提示词版本号 |
| sentence_id | VARCHAR | 是 | 句子ID |
| parse_status | VARCHAR | 是 | 解析状态（success/failed等） |
| retry_count | INTEGER | 是 | 重试次数 |
| error_type | VARCHAR | 否 | 错误类型（如果解析失败） |
| llm_output | JSON | 否 | LLM原始输出（JSON格式） |

### 7. aspect_sentiment_valid（验证后的aspect和情感表）

存储经过验证和规范化的aspect、情感和问题信息。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| sentence_id | VARCHAR | 是 | 句子ID |
| review_pk | VARCHAR | 是 | 评论主键 |
| parent_asin | VARCHAR | 是 | 父产品ASIN |
| timestamp | INTEGER | 是 | 评论时间戳 |
| rating | DOUBLE | 是 | 评分 |
| aspect_raw | VARCHAR | 是 | 原始aspect文本 |
| aspect_norm | VARCHAR | 是 | 规范化后的aspect |
| sentiment | VARCHAR | 是 | 情感标签（positive/negative/neutral） |
| sentiment_score | DOUBLE | 否 | 情感分数 |
| issue_raw | VARCHAR | 是 | 原始问题描述 |
| issue_norm | VARCHAR | 是 | 规范化后的问题描述 |
| evidence_text | VARCHAR | 是 | 证据文本 |
| validity_label | VARCHAR | 是 | 有效性标签（VALID/INVALID） |
| quality_flags | VARCHAR | 否 | 质量标记 |

### 8. extraction_issues（提取问题表）

记录提取过程中发现的问题和异常。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| sentence_id | VARCHAR | 是 | 句子ID |
| issue_type | VARCHAR | 是 | 问题类型 |
| details | VARCHAR | 否 | 问题详情 |
| needs_recheck | BOOLEAN | 否 | 是否需要重新检查 |

### 9. issue_clusters（问题簇表）

存储问题聚类结果，记录每个句子所属的簇。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| embedding_model | VARCHAR | 是 | 使用的embedding模型名称 |
| clustering_config_id | VARCHAR | 是 | 聚类配置ID（JSON字符串） |
| aspect_norm | VARCHAR | 是 | 规范化后的aspect |
| cluster_id | VARCHAR | 是 | 簇ID（"noise"表示噪声点） |
| sentence_id | VARCHAR | 是 | 句子ID |
| cluster_key_text | VARCHAR | 是 | 簇关键文本（结构化输入） |
| issue_norm | VARCHAR | 是 | 规范化后的问题描述 |
| sentiment | VARCHAR | 是 | 情感标签 |
| is_noise | BOOLEAN | 否 | 是否为噪声点 |
| cluster_embedding | FLOAT[N] | 否 | 聚类时使用的向量（E_issue向量，N为向量维度，由embedding_tool自动检测，支持降维） |

### 10. cluster_stats（簇统计表）

存储每个簇的统计信息和质量指标。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| aspect_norm | VARCHAR | 是 | 规范化后的aspect |
| cluster_id | VARCHAR | 是 | 簇ID |
| cluster_size | INTEGER | 是 | 簇大小（包含的样本数） |
| neg_ratio | DOUBLE | 是 | 负面情感比例 |
| intra_cluster_distance | DOUBLE | 否 | 簇内平均距离 |
| inter_cluster_distance | DOUBLE | 否 | 簇间平均距离 |
| separation_ratio | DOUBLE | 否 | 分离度比率 |
| cohesion | DOUBLE | 否 | 簇内聚度 |
| cluster_confidence | DOUBLE | 否 | 簇置信度 |
| sentiment_consistency | DOUBLE | 否 | 情感一致性 |
| recent_trend | JSON | 否 | 近期趋势（JSON格式） |
| top_terms | JSON | 否 | 高频词（JSON格式） |
| representative_sentence_ids | JSON | 是 | 代表样本ID列表（JSON数组） |

### 11. cluster_reports（簇报告表）

存储LLM生成的簇洞察报告，包含命名、摘要、建议等。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| llm_model | VARCHAR | 是 | 使用的LLM模型名称 |
| prompt_version | VARCHAR | 是 | 提示词版本号 |
| aspect_norm | VARCHAR | 是 | 规范化后的aspect |
| cluster_id | VARCHAR | 是 | 簇ID |
| cluster_name | VARCHAR | 是 | 簇名称（LLM生成） |
| summary | VARCHAR | 是 | 簇摘要（2-3句描述） |
| priority | VARCHAR | 是 | 优先级（high/medium/low） |
| evidence_items | JSON | 是 | 证据条目列表（JSON数组） |
| action_items | JSON | 是 | 可执行建议列表（JSON数组） |
| risks_and_assumptions | JSON | 否 | 风险和假设列表（JSON数组） |
| confidence | DOUBLE | 否 | 置信度（0-1） |

### 12. evaluation_metrics（评估指标表）

存储各种评估指标和性能指标。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| metric_name | VARCHAR | 是 | 指标名称 |
| metric_value | DOUBLE | 是 | 指标值 |
| metric_scope | VARCHAR | 否 | 指标作用域 |
| notes | VARCHAR | 否 | 备注说明 |

### 13. run_log（运行日志表）

记录流水线各步骤的执行日志。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| step_name | VARCHAR | 是 | 步骤名称 |
| status | VARCHAR | 是 | 状态（success/failed/running等） |
| input_rows | INTEGER | 否 | 输入行数 |
| output_rows | INTEGER | 否 | 输出行数 |
| error_rows | INTEGER | 否 | 错误行数 |
| started_at | TIMESTAMP | 是 | 开始时间 |
| finished_at | TIMESTAMP | 否 | 结束时间 |
| message | VARCHAR | 否 | 日志消息 |

## 传统方法表（Traditional Baseline，无LLM）

传统方法表用于不使用LLM的baseline流水线，与主线表一一对应，使用`_traditional`后缀区分。

### 14. aspect_sentiment_raw_traditional（传统原始抽取表）

存储传统NLP方法抽取的aspect和sentiment结果。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| sentence_id | VARCHAR | 是 | 句子ID |
| extract_method | VARCHAR | 是 | 抽取方法（LEXICON_RULE/WEAK_SUPERVISED/HYBRID） |
| aspect_raw | VARCHAR | 是 | 原始aspect |
| issue_raw | VARCHAR | 是 | 原始issue |
| sentiment | VARCHAR | 是 | 情感（positive/negative/neutral） |
| sentiment_score | DOUBLE | 否 | 情感分数 |
| evidence_text | VARCHAR | 是 | 证据文本（可定位） |
| debug_features | JSON | 否 | 调试信息（命中的规则/模板/关键词） |

### 15. aspect_sentiment_valid_traditional（传统验证表）

存储经过校验和归一化的传统抽取结果，结构与`aspect_sentiment_valid`一致。

### 16. extraction_issues_traditional（传统提取问题表）

存储传统抽取过程中的问题和错误，结构与`extraction_issues`一致。

### 17. issue_clusters_traditional（传统问题簇表）

存储传统方法的聚类归属结果，结构与`issue_clusters`一致，包含`cluster_embedding`向量字段。

### 18. cluster_stats_traditional（传统簇统计表）

存储传统方法的簇统计信息，结构与`cluster_stats`一致。

### 19. cluster_reports_traditional（传统簇报告表）

存储传统方法的模板化洞察报告。

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| run_id | VARCHAR | 是 | 运行ID |
| pipeline_version | VARCHAR | 是 | 流水线版本号 |
| data_slice_id | VARCHAR | 是 | 数据切片ID |
| created_at | TIMESTAMP | 是 | 记录创建时间 |
| method_version | VARCHAR | 是 | 传统方法版本号 |
| aspect_norm | VARCHAR | 是 | 归一化aspect |
| cluster_id | VARCHAR | 是 | 簇ID |
| cluster_name | VARCHAR | 是 | 簇名称 |
| summary | VARCHAR | 是 | 摘要 |
| priority | VARCHAR | 是 | 优先级（high/medium/low） |
| priority_rationale | VARCHAR | 否 | 优先级依据 |
| evidence_items | JSON | 是 | 证据条目列表 |
| action_items | JSON | 是 | 建议条目列表 |
| confidence | DOUBLE | 否 | 置信度（基于规则估算） |

## 表创建

所有表（包括传统方法表）通过`TableManager.create_all_tables()`统一创建。

