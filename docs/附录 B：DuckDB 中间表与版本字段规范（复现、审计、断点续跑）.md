# 附录 B：DuckDB 中间表与版本字段规范（复现、审计、断点续跑）

## B.0 全局约定（所有中间表通用）

### B.0.1 命名与字段风格

* 表名：`snake_case`，语义清晰（如 `review_sentences`, `aspect_sentiment_valid`）
* 字段名：`snake_case`，避免与源表同名字段混淆

  * 例如：`meta.title` → `product_title`
  * `reviews.title` → `review_title`

### B.0.2 主键与可追溯性（必须具备）

由于源表未提供显式主键，需在流水线中构造稳定键：

* **`review_pk`（评论主键）**：建议定义为可复现的“确定性键”，基于评论的稳定组合字段（如 `parent_asin + user_id + timestamp + title + text` 的哈希/摘要）。
* **`sentence_id`（句子主键）**：建议为 `review_pk + sentence_index` 的确定性键（同一评论同一句子位置应稳定）。

> 要求：所有下游表必须能通过 `sentence_id → review_pk → reviews` 回溯到原始评论内容与元信息。

### B.0.3 运行版本字段（建议所有表都带）

为保证复现性、可审计性、支持断点续跑，建议所有中间表至少包含以下字段（即使有些表只存一次，也建议记录）：

| 字段                   | 类型        | 说明                        |
| -------------------- | --------- | ------------------------- |
| `run_id`             | string    | 一次完整流水线运行的唯一标识            |
| `pipeline_version`   | string    | 管道版本（如 v1.1）              |
| `data_slice_id`      | string    | 数据切片标识（类目/商品/时间窗/过滤条件的摘要） |
| `created_at`         | timestamp | 记录生成时间                    |
| `source_snapshot_at` | timestamp | 源数据快照时间（或提取时间），用于解释数据变化   |

### B.0.4 模型与 Prompt 版本字段（涉及 LLM/Embedding 的表必须带）

| 字段                     | 类型     | 说明                             |
| ---------------------- | ------ | ------------------------------ |
| `llm_model`            | string | 使用的 LLM 型号/版本                  |
| `prompt_version`       | string | Prompt 模板版本                    |
| `embedding_model`      | string | Embedding 模型名称/版本              |
| `clustering_config_id` | string | 聚类参数配置ID（min_cluster_size 等摘要） |

---

## B.1 表规范：`selected_reviews`（实验样本固定表）

### 目的

固定本次实验的评论样本范围，支持重复运行与消融实验一致对比。

### 必备字段

| 字段                                                          | 类型     | 必填 | 说明               |
| ----------------------------------------------------------- | ------ | -: | ---------------- |
| `run_id`, `pipeline_version`, `data_slice_id`, `created_at` | 见 B.0  |  是 | 版本化字段            |
| `review_pk`                                                 | string |  是 | 构造主键             |
| `parent_asin`                                               | string |  是 | 连接键              |
| `asin`                                                      | string |  否 | 变体ID（可保留）        |
| `user_id`                                                   | string |  否 | 可选               |
| `timestamp`                                                 | int    |  是 | 评论时间             |
| `rating`                                                    | float  |  是 | 评分               |
| `verified_purchase`                                         | bool   |  否 | 数据质量字段           |
| `helpful_vote`                                              | int    |  否 | 数据质量字段           |
| `review_title`                                              | string |  否 | 来自 reviews.title |
| `review_text`                                               | string |  是 | 来自 reviews.text  |
| `main_category`                                             | string |  否 | join meta 后得到    |
| `product_title`                                             | string |  否 | join meta 后得到    |

### 建议索引/查询习惯

* 常用过滤：`parent_asin`, `main_category`, `timestamp`
* 常用 join：`review_pk`

---

## B.2 表规范：`meta_context`（元数据上下文表：截断/摘要后）

### 目的

为 LLM 提供稳定、短小、低成本的商品语境，避免 `features` 超长导致 token 溢出。

### 必备字段

| 字段                                                          | 类型     | 必填 | 说明                                      |
| ----------------------------------------------------------- | ------ | -: | --------------------------------------- |
| `run_id`, `pipeline_version`, `data_slice_id`, `created_at` |        |  是 | 版本化                                     |
| `parent_asin`                                               | string |  是 | 主键之一                                    |
| `product_title`                                             | string |  是 | 来自 meta.title                           |
| `main_category`                                             | string |  否 | 来自 meta.main_category                   |
| `features_short`                                            | string |  否 | features 截断/摘要                          |
| `description_short`                                         | string |  否 | 可选                                      |
| `details_short`                                             | string |  否 | 可选                                      |
| `context_version`                                           | string |  是 | 上下文生成策略版本（如“truncate_512”或“summary_v2”） |

---

## B.3 表规范：`normalized_reviews`（清洗后的评论表）

### 目的

将原始评论文本规范化，为分句与过滤做准备。

### 必备字段

| 字段                                   | 类型                     | 必填 | 说明                       |
| ------------------------------------ | ---------------------- | -: | ------------------------ |
| 版本字段（run_id等）                        |                        |  是 |                          |
| `review_pk`                          | string                 |  是 |                          |
| `parent_asin`, `timestamp`, `rating` |                        |  是 | 继承自 selected_reviews     |
| `clean_text`                         | string                 |  是 | 合并并清洗后的文本（例如 title+text） |
| `cleaning_flags`                     | array(string) / string |  否 | 记录清洗动作（去HTML、去重等）        |

---

## B.4 表规范：`review_sentences`（句子表：含上下文窗口）

### 目的

形成 LLM 抽取的最关键输入单元，**同时保存目标句与上下文**以解决指代与转折问题。

### 必备字段

| 字段                                  | 类型     | 必填 | 说明        |
| ----------------------------------- | ------ | -: | --------- |
| 版本字段                                |        |  是 |           |
| `sentence_id`                       | string |  是 | 主键        |
| `review_pk`                         | string |  是 | 外键回溯      |
| `parent_asin`                       | string |  是 |           |
| `timestamp`                         | int    |  是 |           |
| `rating`                            | float  |  是 |           |
| `verified_purchase`, `helpful_vote` |        |  否 |           |
| `sentence_index`                    | int    |  是 | 句子在评论中的顺序 |
| `target_sentence`                   | string |  是 | 抽取对象      |
| `prev_sentence`                     | string |  否 | 可选        |
| `next_sentence`                     | string |  否 | 可选        |
| `context_text`                      | string |  是 | 上下文窗口文本   |

---

## B.5 表规范：`opinion_candidates`（观点候选过滤结果表）

### 目的

控制 LLM 调用规模，降低成本与耗时。

### 必备字段

| 字段                | 类型     | 必填 | 说明                              |
| ----------------- | ------ | -: | ------------------------------- |
| 版本字段              |        |  是 |                                 |
| `sentence_id`     | string |  是 | 主键                              |
| `is_candidate`    | bool   |  是 | 是否进入 LLM 抽取                     |
| `filter_reason`   | string |  否 | 规则命中原因（用于调参）                    |
| `priority_weight` | float  |  否 | 优先处理权重（如 helpful_vote 高/强情感句优先） |

---

## B.6 表规范：`aspect_sentiment_raw`（LLM 原始抽取结果表）

### 目的

保留 LLM 直接输出，便于审计与错误分析；不直接作为下游可信输入。

### 必备字段

| 字段             | 类型            | 必填 | 说明                                  |
| -------------- | ------------- | -: | ----------------------------------- |
| 版本字段           |               |  是 | 需包含 llm_model、prompt_version        |
| `sentence_id`  | string        |  是 | 主键                                  |
| `parse_status` | string        |  是 | `SUCCESS/FAIL`                      |
| `retry_count`  | int           |  是 | 重试次数                                |
| `error_type`   | string        |  否 | 失败类型（如 JSON_PARSE / TOOL_CALL_FAIL） |
| `llm_output`   | JSON / string |  是 | 原始结构化输出（建议用 JSON 类型存储）              |

> 建议保留输入摘要字段（可选）：如 `input_hash`、`context_version`，用于确认同一输入不重复调用。

---

## B.7 表规范：`aspect_sentiment_valid`（校验后事实表：唯一可信输入）

### 目的

输出标准化、可用于聚类与统计的“方面级事实记录”（一行一个 aspect）。

### 必备字段

| 字段                | 类型                     | 必填 | 说明                                       |
| ----------------- | ---------------------- | -: | ---------------------------------------- |
| 版本字段              |                        |  是 |                                          |
| `sentence_id`     | string                 |  是 | 外键                                       |
| `review_pk`       | string                 |  是 | 外键                                       |
| `parent_asin`     | string                 |  是 |                                          |
| `timestamp`       | int                    |  是 |                                          |
| `rating`          | float                  |  是 |                                          |
| `aspect_raw`      | string                 |  是 | LLM 原始 aspect                            |
| `aspect_norm`     | string                 |  是 | 规范化后 aspect（同义归一结果）                      |
| `sentiment`       | string                 |  是 | pos/neg/neu（或 positive/negative/neutral） |
| `sentiment_score` | float                  |  否 |                                          |
| `issue_raw`       | string                 |  是 | LLM 原始 issue                             |
| `issue_norm`      | string                 |  是 | 规范化后 issue（清洗/裁剪/去噪）                     |
| `evidence_text`   | string                 |  是 | 必须可定位                                    |
| `validity_label`  | string                 |  是 | 枚举：`VALID` / `NOISE` / `INVALID`         |
| `quality_flags`   | string / array(string) |  否 | 冲突/低置信/证据弱等标记                            |

### `validity_label` 判定原则（建议写入实现说明）

* `VALID`：证据可定位、字段完整、非泛化噪声
* `NOISE`：过泛化/无意义（会污染聚类），例如 aspect/issue 过泛且无修饰
* `INVALID`：结构错误、证据不可定位、明显跑题或不合规则

---

## B.8 表规范：`extraction_issues`（抽取问题与错误分析表）

### 目的

记录被过滤/冲突/失败样本，支撑错误分析与复核队列。

### 必备字段

| 字段              | 类型     | 必填 | 说明                                                                         |
| --------------- | ------ | -: | -------------------------------------------------------------------------- |
| 版本字段            |        |  是 |                                                                            |
| `sentence_id`   | string |  是 |                                                                            |
| `issue_type`    | string |  是 | 如 `EVIDENCE_NOT_FOUND / LOW_SPECIFICITY / SENTIMENT_CONFLICT / PARSE_FAIL` |
| `details`       | string |  否 | 补充说明                                                                       |
| `needs_recheck` | bool   |  否 | 是否进入 LLM 复核队列                                                              |

---

## B.9 表规范：`issue_clusters`（聚类归属表）

### 目的

记录每条 VALID 事实记录所属簇，支撑簇统计与洞察生成。

### 必备字段

| 字段                 | 类型           | 必填 | 说明                                            |
| ------------------ | ------------ | -: | --------------------------------------------- |
| 版本字段               |              |  是 | 需包含 embedding_model、clustering_config_id      |
| `aspect_norm`      | string       |  是 | 聚类桶维度                                         |
| `cluster_id`       | string / int |  是 |                                               |
| `sentence_id`      | string       |  是 |                                               |
| `cluster_key_text` | string       |  是 | **用于向量化的文本**：`aspect_norm + " " + issue_norm` |
| `issue_norm`       | string       |  是 |                                               |
| `sentiment`        | string       |  是 |                                               |
| `is_noise`         | bool         |  否 | 聚类算法噪声点（如 HDBSCAN）                            |

---

## B.10 表规范：`cluster_stats`（簇统计表）

### 目的

为 Top 簇排序、优先级计算、趋势分析提供数值依据。

### 必备字段

| 字段                            | 类型            | 必填 | 说明                         |
| ----------------------------- | ------------- | -: | -------------------------- |
| 版本字段                          |               |  是 |                            |
| `aspect_norm`                 | string        |  是 |                            |
| `cluster_id`                  | string / int  |  是 |                            |
| `cluster_size`                | int           |  是 |                            |
| `neg_ratio`                   | float         |  是 | 负面占比                       |
| `recent_trend`                | JSON / string |  否 | 趋势统计摘要（如最近30天变化）           |
| `top_terms`                   | JSON / string |  否 | 关键词摘要（可选）                  |
| `representative_sentence_ids` | JSON / string |  是 | 代表样本 sentence_id 列表（供洞察生成） |

---

## B.11 表规范：`cluster_reports`（簇洞察表：LLM 输出）

### 目的

存储最终可展示的簇洞察与建议，是报告核心来源。

### 必备字段

| 字段                      | 类型            | 必填 | 说明                                       |
| ----------------------- | ------------- | -: | ---------------------------------------- |
| 版本字段                    |               |  是 | 需包含 llm_model、prompt_version（洞察Prompt版本） |
| `aspect_norm`           | string        |  是 |                                          |
| `cluster_id`            | string / int  |  是 |                                          |
| `cluster_name`          | string        |  是 |                                          |
| `summary`               | string        |  是 |                                          |
| `priority`              | string        |  是 | high/medium/low                          |
| `evidence_items`        | JSON / string |  是 | 必须可回溯 sentence_id                        |
| `action_items`          | JSON / string |  是 | 结构化建议                                    |
| `risks_and_assumptions` | JSON / string |  否 |                                          |
| `confidence`            | float         |  否 |                                          |

---

## B.12 表规范：`evaluation_metrics`（评估指标表）

### 目的

存储自动评估结果与人工评估摘要，支撑报告“实验”部分。

### 必备字段

| 字段             | 类型     | 必填 | 说明                              |
| -------------- | ------ | -: | ------------------------------- |
| 版本字段           |        |  是 |                                 |
| `metric_name`  | string |  是 | 如 `coverage_rate`, `silhouette` |
| `metric_value` | float  |  是 |                                 |
| `metric_scope` | string |  否 | 如 `overall`/`by_aspect`         |
| `notes`        | string |  否 | 解释口径与计算范围                       |

---

## B.13 建议新增：`run_log`（运行日志表，强烈推荐）

### 目的

让系统“可运维、可复盘”，避免黑盒。

| 字段            | 类型        | 必填 | 说明                                |
| ------------- | --------- | -: | --------------------------------- |
| `run_id`      | string    |  是 |                                   |
| `step_name`   | string    |  是 | 如 `sentence_build`, `llm_extract` |
| `status`      | string    |  是 | `STARTED/SUCCESS/FAILED`          |
| `input_rows`  | int       |  否 | 输入量                               |
| `output_rows` | int       |  否 | 输出量                               |
| `error_rows`  | int       |  否 | 失败量                               |
| `started_at`  | timestamp |  是 |                                   |
| `finished_at` | timestamp |  否 |                                   |
| `message`     | string    |  否 | 简短日志                              |

---

## B.14 断点续跑与增量策略（写进实现说明即可）

* 以 `sentence_id` 为最小处理单元：

  * `aspect_sentiment_raw` 中已存在 `sentence_id` 且 `parse_status=SUCCESS` → 不再重复抽取
* 复核与重试：

  * `aspect_sentiment_raw.parse_status=FAIL` 或 `extraction_issues.needs_recheck=true` → 进入重试队列
* 版本隔离：

  * 不同 `run_id` 的结果可共存；对比实验仅需切换 `run_id/data_slice_id`

---

