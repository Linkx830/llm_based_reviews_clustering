
# 附录 A：LLM 结构化输出 Schema 规范（Extraction & Insight）

## A.0 设计原则（必须遵守）

1. **结构化强约束**：优先使用支持 Function Calling / Tool Call 的模式输出结构化字段；否则使用 LangChain 的结构化解析器（Pydantic/StructuredOutputParser）。
2. **可审计**：LLM 输出必须可回溯到输入（尤其是 evidence），禁止“无证据推断”。
3. **最小必要信息**：不输出冗长推理过程；输出以**可计算字段**为主，便于聚类、统计与报告生成。
4. **面向下游**：抽取输出的字段要能直接支持 `ExtractionJudgeAgent` 标准化与 `IssueClusterAgent` 聚类。

---

## A.1 结构化抽取：AspectSentimentExtractorAgent 输出规范

### A.1.1 输入对象（供 LLM 使用的业务输入，不是输出）

> 该输入由 Orchestrator 组织，主要来自 `review_sentences` + `meta_context`。

| 字段                | 类型     | 必填 | 说明                              |
| ----------------- | ------ | -: | ------------------------------- |
| `sentence_id`     | string |  是 | 句子唯一 ID（用于落库与去重）                |
| `parent_asin`     | string |  是 | 商品父ID，用于关联元数据语境                 |
| `target_sentence` | string |  是 | **唯一抽取对象**：只对这一句做抽取             |
| `context_text`    | string |  是 | 上下文窗口（如前一句+当前句+后一句），用于指代消解与转折理解 |
| `product_title`   | string |  否 | 来自 meta.title（建议提供）             |
| `main_category`   | string |  否 | 来自 meta.main_category（建议提供）     |
| `features_short`  | string |  否 | meta.features 摘要/截断版本（建议提供）     |
| `details_short`   | string |  否 | meta.details 摘要/截断版本（可选）        |

**强制说明（Prompt 约束必须包含）**

* `context_text` 仅用于理解，不得抽取其中非 `target_sentence` 的事实。
* evidence 必须来自 `target_sentence` 可定位片段。

---

### A.1.2 输出对象（LLM 结构化输出 Schema）

#### 顶层字段

| 字段                    | 类型            | 必填 | 约束/说明                                      |
| --------------------- | ------------- | -: | ------------------------------------------ |
| `sentence_id`         | string        |  是 | 必须原样返回，用于对齐                                |
| `has_opinion`         | boolean       |  是 | 该句是否包含可抽取的观点/评价/问题；如为 false，`aspects` 必须为空 |
| `aspects`             | array         |  是 | 方面抽取结果列表；`has_opinion=false` 时必须为 `[]`     |
| `language`            | string        |  否 | `en/zh/auto`；可用于后续分析（可选）                   |
| `extraction_warnings` | array(string) |  否 | 非致命问题提示，用于后续质量分析（见 A.1.4）                  |

#### `aspects[]` 元素字段（每条代表“一个方面的评价/问题”）

| 字段                    | 类型      | 必填 | 约束/说明                                                     |
| --------------------- | ------- | -: | --------------------------------------------------------- |
| `aspect`              | string  |  是 | **名词短语**，表达组件/维度；长度建议 ≤ 5 词；不得为纯代词（it/this）               |
| `sentiment`           | string  |  是 | 枚举：`positive` / `negative` / `neutral`                    |
| `sentiment_score`     | number  |  否 | 建议范围：[-1, 1]；与 sentiment 方向一致（负为负面）                       |
| `issue`               | string  |  是 | 对方面的具体描述，**优先短语化**；长度建议 ≤ 12 词；避免“too good/very nice”这种空泛 |
| `evidence_text`       | string  |  是 | 必须是 `target_sentence` 的**子串**或可直接定位的片段                    |
| `evidence_start_char` | integer |  否 | evidence 在 `target_sentence` 中的起始字符索引（推荐提供，便于自动校验）        |
| `evidence_end_char`   | integer |  否 | evidence 在 `target_sentence` 中的结束字符索引（推荐提供）               |
| `polarity_target`     | string  |  否 | 可选补充：说明 sentiment 指向的对象（一般等于 aspect）                      |
| `confidence`          | number  |  否 | 建议范围：[0,1]；低置信可在 Judge 中触发更严格过滤                           |

---

### A.1.3 关键约束（必须由 Judge 自动校验）

1. **Evidence 可定位性（硬约束）**

   * `evidence_text` 必须能在 `target_sentence` 中匹配定位。
   * 如果提供 `evidence_start_char/end_char`，需与匹配结果一致。
   * 不满足则该 aspect 记录应被标记为 `INVALID`（见附录 B 的 validity_label）。

2. **对象范围约束（硬约束）**

   * 输出不得引用 `context_text` 的其他句作为证据。
   * 若模型需要上下文才能确定指代，应在 aspect/issue 中明确实体（如“screen brightness”而非“it brightness”）。

3. **空泛/噪声约束（硬约束）**

   * `aspect` 或 `issue` 若为极泛化词且无修饰信息，应触发 `NOISE/INVALID`：

     * aspect 示例：`product`, `item`, `thing`, `it`, `quality`（无上下文限定）
     * issue 示例：`good`, `bad`, `nice`, `works`, `not good`（缺少对象/维度）

4. **多方面上限（软约束）**

   * 单句方面数量建议 ≤ 3。超过时应触发 `extraction_warnings`（如 `TOO_MANY_ASPECTS`），以便后续治理。

---

### A.1.4 `extraction_warnings` 推荐枚举（用于诊断，不直接判死刑）

| 枚举值                 | 含义                | 典型处理                |
| ------------------- | ----------------- | ------------------- |
| `PRONOUN_AMBIGUOUS` | 存在 it/this 等指代不明  | Judge 可标记低置信或进入复核队列 |
| `LOW_SPECIFICITY`   | issue 过于空泛        | Judge 可能转 NOISE     |
| `TOO_MANY_ASPECTS`  | 单句抽取方面过多          | 可只保留置信度最高的前 N 个     |
| `WEAK_EVIDENCE`     | evidence 过短或不支撑结论 | Judge 标记低置信         |
| `CONTEXT_DEPENDENT` | 理解明显依赖上下文         | Judge 可优先保留但标记需要上下文 |

---

## A.2 簇洞察：ClusterInsightAgent 输出规范（命名/摘要/建议）

### A.2.1 输入对象（供 LLM 生成洞察）

> 输入来自 `cluster_stats` + 从簇中抽取的代表样本（建议包含句子ID和原句）。

| 字段                       | 类型            | 必填 | 说明                                       |
| ------------------------ | ------------- | -: | ---------------------------------------- |
| `aspect_norm`            | string        |  是 | 已规范化的方面名称（来自 Judge 输出）                   |
| `cluster_id`             | string / int  |  是 | 簇ID                                      |
| `cluster_size`           | integer       |  是 | 簇内样本数                                    |
| `neg_ratio`              | number        |  是 | 负面占比（0~1）                                |
| `recent_trend`           | object        |  否 | 趋势统计（如最近30天占比变化）                         |
| `representative_samples` | array(object) |  是 | 代表样本列表（建议 10–20 条）                       |
| `product_context`        | object        |  否 | 可选：product_title/features_short 等，避免建议跑题 |

**`representative_samples[]` 建议字段**

* `sentence_id`
* `target_sentence`
* `issue_norm`
* `sentiment`
* `helpful_vote`（如有）
* `timestamp`（如要做趋势说明）

---

### A.2.2 输出对象（LLM 结构化输出 Schema）

#### 顶层字段

| 字段                      | 类型            | 必填 | 约束/说明                                       |
| ----------------------- | ------------- | -: | ------------------------------------------- |
| `aspect_norm`           | string        |  是 | 必须原样返回                                      |
| `cluster_id`            | string / int  |  是 | 必须原样返回                                      |
| `cluster_name`          | string        |  是 | 一句话命名；必须体现 aspect；建议 ≤ 12 词（英文）或 ≤ 20 字（中文） |
| `summary`               | string        |  是 | 2–3 句现象总结：描述“发生了什么 + 对用户影响”                 |
| `priority`              | string        |  是 | 枚举：`high` / `medium` / `low`                |
| `priority_rationale`    | string        |  否 | 简短说明优先级依据（规模/负面率/趋势）                        |
| `evidence_items`        | array(object) |  是 | 证据条目（必须可回溯到输入样本）                            |
| `action_items`          | array(object) |  是 | 可执行建议条目（建议 3 条）                             |
| `risks_and_assumptions` | array(string) |  否 | 风险/假设，避免过度确定性                               |
| `confidence`            | number        |  否 | 0~1，表示洞察可靠程度                                |

#### `evidence_items[]` 字段

| 字段                   | 类型     | 必填 | 约束/说明                                |
| -------------------- | ------ | -: | ------------------------------------ |
| `sentence_id`        | string |  是 | 必须来自输入 `representative_samples`      |
| `quote`              | string |  是 | 必须与对应样本的 `target_sentence` 一致或为其子串   |
| `why_representative` | string |  否 | 简短说明为何典型（如“高频”“描述具体”“helpful_vote高”） |

> **硬约束**：不得出现输入样本中不存在的 `sentence_id`；不得生成“凭空证据句”。

#### `action_items[]` 字段（建议结构化到“可执行”）

| 字段                | 类型     | 必填 | 约束/说明                                                       |
| ----------------- | ------ | -: | ----------------------------------------------------------- |
| `action`          | string |  是 | 建议动作本体（具体、可落地）                                              |
| `owner_team`      | string |  否 | 建议责任方：`product/design/quality/logistics/customer_service` 等 |
| `expected_impact` | string |  否 | 预期影响（如“降低退货率”“提升满意度”）                                       |
| `validation_plan` | string |  否 | 如何验证有效（如A/B、抽检、实验）                                          |
| `urgency`         | string |  否 | `high/medium/low`（可与 priority 相关但不等同）                       |

---

### A.2.3 洞察输出的强约束规则（必须执行）

1. **证据闭环**：所有 `evidence_items` 必须能回溯到输入样本；否则该 cluster_report 视为不合格。
2. **方面一致性**：`cluster_name/summary/action_items` 不得脱离 `aspect_norm`，不得跨组件乱提建议。
3. **建议可验证**：至少 1 条 action 必须包含明确的 `validation_plan`（让报告更“科学/工程化”）。
4. **不过度确定**：当 `cluster_size` 很小或证据弱时，应在 `risks_and_assumptions` 中标注不确定性。

---
