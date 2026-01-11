# agents/ 模块说明

## 职责

各Agent实现模块，每个Agent职责单一、输入输出清晰。

## Agent列表

1. **DataSelectorAgent**: 数据切片与实验样本固定
   - 输入：reviews, meta
   - 输出：selected_reviews
   - 注意：使用DuckDB的`?`占位符和位置参数列表进行查询

2. **MetaContextAgent**: 元数据上下文构建（截断/摘要）
   - 输入：meta, selected_reviews
   - 输出：meta_context

3. **PreprocessAgent**: 评论级清洗与文本规范化
   - 输入：selected_reviews
   - 输出：normalized_reviews

4. **SentenceBuilderAgent**: 句子构建 + 上下文窗口
   - 输入：normalized_reviews
   - 输出：review_sentences

5. **OpinionCandidateFilterAgent**: 观点候选句过滤（成本控制）
   - 输入：review_sentences
   - 输出：opinion_candidates
   - 功能：
     - 使用评分系统识别观点候选句，避免遗漏鼓励、批评、建议等有效观点
     - 支持多维度评分：强评价关键词、建议类、比较类、问题描述类、功能评价类、经验分享类
     - 使用正则表达式模式匹配识别常见观点表达结构
     - 输出观点分数（priority_weight）用于优先级排序
   - 配置参数：
     - `min_length`: 最小句长（默认10）
     - `max_length`: 最大句长（默认500）
     - `score_threshold`: 观点分数阈值（默认2.0），达到此分数才被认为是候选句
   - 评分规则：
     - 强评价关键词（如good/bad/love/hate）：权重2.0，最多4分
     - 建议类关键词（如should/recommend/suggest）：权重1.5，最多3分
     - 比较类关键词（如better/worse/compared）：权重1.5，最多3分
     - 问题描述类关键词（如problem/issue/broken）：权重1.5，最多3分
     - 功能评价类关键词（如works/performs/functions）：权重1.0，最多2分
     - 经验分享类关键词（如after/since/used/tried）：权重1.0，最多2分
     - 否定词+程度词组合：权重1.5
     - 模式匹配（正则表达式）：权重1.0-1.5
     - 其他特征（感叹号、第一人称、疑问句等）：权重0.3-0.5

6. **AspectSentimentExtractorAgent**: LLM结构化抽取
   - 输入：review_sentences, opinion_candidates, meta_context
   - 输出：aspect_sentiment_raw
   - 注意：JOIN opinion_candidates 时必须同时匹配 sentence_id 和 run_id，避免跨 run_id 数据污染
   - 配置参数：
     - `llm_wrapper`: LLMWrapper实例（从配置中传入，使用`models.extraction_llm`配置）
     - `prompt_version`: Prompt版本（默认v1.0，从configs/prompts/extraction/加载）
     - `max_retries`: 最大重试次数（默认3）
   - 模型配置：
     - 在`configs/runs/*.yaml`中通过`models.extraction_llm`配置
     - 推荐使用较小但精确的模型（如`qwen3:8b`），temperature=0.0
     - 如果未指定`extraction_llm`，则使用`models.llm_provider`和`models.llm_model`作为默认值

7. **ExtractionJudgeAgent**: 抽取校验、归一、噪声处理
   - 输入：aspect_sentiment_raw, review_sentences
   - 输出：aspect_sentiment_valid, extraction_issues
   - 功能：
     - **增强的aspect归一化**：支持多种匹配策略（按优先级）：
       1. **精确匹配**：完全匹配同义词表
       2. **部分匹配**：aspect包含同义词表中的词（匹配度≥50%）
       3. **模糊匹配**：使用SequenceMatcher计算编辑距离相似度（阈值≥0.75）
       4. **关键词提取匹配**：使用Jaccard相似度匹配核心词（阈值≥0.5）
       5. **语义相似度匹配**：使用embedding计算余弦相似度（可选，阈值≥0.7）
       6. **LLM辅助匹配**：使用大模型判断最合适的归一化aspect（可选，作为最后手段）
     - 噪声识别与过滤
     - 证据可定位性校验
   - 配置参数：
     - `embedding_tool`: Embedding工具（可选，用于语义匹配）
     - `use_semantic_matching`: 是否启用语义相似度匹配（默认False）
     - `semantic_threshold`: 语义相似度阈值（默认0.7）
     - `llm_wrapper`: LLM包装器（可选，用于LLM辅助匹配）
     - `use_llm_matching`: 是否启用LLM辅助匹配（默认False）
     - `llm_confidence_threshold`: LLM匹配的置信度阈值（默认0.5）
     - 使用建议：
     - 默认情况下使用前4种策略（无需embedding/LLM），可处理大部分情况
     - 如果大模型输出的aspect经常不在词表中，可启用语义匹配（需要在配置中设置`judge.use_semantic_matching=true`）
     - 如果词表太少导致匹配缺失，可启用LLM辅助匹配（需要在配置中设置`judge.use_llm_matching=true`）
     - LLM匹配会调用本地部署的大模型，从词表中选择最合适的aspect，或判断是否为噪声词
     - **LLM有权创建新的aspect**：当词表不足时，LLM可以创建新的标准aspect并自动添加到词表中
     - LLM创建的新aspect会：
       - 自动添加到内存中的词表（本次运行有效）
       - 记录在返回结果中（`new_aspects`和`dynamic_synonyms`）
       - 可以通过`normalizer.get_new_aspects()`和`normalizer.get_dynamic_synonyms()`获取
     - LLM匹配结果会被缓存，避免重复调用

8. **IssueClusterAgent**: 两阶段聚类（基于docs/聚类规范.md）
     - 输入：aspect_sentiment_valid
     - 输出：issue_clusters, cluster_stats
     - 聚类流程（基于新聚类规范）：
       - **阶段A：结构化输入**：使用模板 `Aspect: {aspect_norm}\nIssue: {issue_norm}`
       - **阶段B：两路向量**：
         - E_issue（主向量）：用于issue聚类，支持instruction-aware embedding
         - E_aspect（辅向量）：用于aspect同义合并
       - **阶段C1：Aspect分桶**：合并同义aspect（cosine相似度阈值默认0.85），按标准aspect分桶
       - **阶段C2：桶内Issue聚类**：每个aspect桶内单独聚类issue
         - 数据量 < 阈值（默认1000）：使用 Agglomerative 聚类
         - 数据量 >= 阈值：使用 HDBSCAN 聚类（自动确定簇数）
         - 使用cosine距离（向量已L2归一化）
       - **阶段D：Reranker边界精修**（可选）：
         - 使用Reranker对embedding近邻进行二次验证
         - 构建reranker图：节点=样本，边=reranker高分对（分数≥阈值）
         - 图聚类：使用连通分量算法进行图聚类
         - 标签重分配：基于reranker图聚类结果重新分配标签
         - 融合策略：优先信任reranker的高置信度连接，但保持原标签的连续性
         - 孤立节点处理：对于没有reranker边的节点，保持原标签
       - **阶段E：簇后处理**：
         - 噪点吸附：将噪声点吸附到最近的簇（相似度≥阈值）
         - 小簇处理：合并小簇或标记为噪声
         - 选择medoid作为代表样本
         - 计算聚类质量指标（簇内距离、簇间距离、分离度、紧密度、置信度等）
     - 配置参数：
       - `embedding_model`: embedding模型名称
       - `embedding_base_url`: Ollama API地址（可选）
       - `embedding_max_workers`: Ollama并发请求数（默认4，建议2-8）
       - `embedding_batch_size`: sentence-transformers批处理大小（默认32）
       - `auto_select_threshold`: 自动选择阈值（默认1000）
       - `use_instruction`: 是否使用instruction-aware embedding（默认True）
       - `issue_instruction`: issue聚类的instruction文本（默认英文instruction）
       - `aspect_instruction`: aspect聚类的instruction文本（默认英文instruction）
       - `aspect_similarity_threshold`: aspect同义合并的相似度阈值（默认0.85）
       - `use_reranker`: 是否启用reranker二次验证（默认False）
       - `reranker_model`: Reranker模型名称（可选）
       - `reranker_base_url`: Reranker API地址（可选，默认使用embedding_base_url）
       - `reranker_llm_wrapper`: LLM包装器（用于LLM-based reranker，可选）
       - `reranker_top_k`: 每个样本的reranker候选数（默认50）
       - `reranker_score_threshold`: Reranker分数阈值（默认0.6）
       - `reranker_max_workers`: Reranker并发数（默认4）
       - `min_cluster_size`: 最小簇大小（默认2，小于此值的簇将被处理）
       - `noise_adsorption_threshold`: 噪点吸附到最近簇的相似度阈值（默认0.7）
       - `small_cluster_merge_threshold`: 小簇合并的相似度阈值（默认0.75）
       - `clustering_config`: 聚类配置（可选）
         - `method`: 可手动指定 "hdbscan" 或 "agglomerative"（覆盖自动选择）
         - `min_cluster_size`: HDBSCAN的最小簇大小（默认自适应：max(10, n*0.005)）
         - `min_samples`: HDBSCAN的最小样本数（默认3）
         - `distance_threshold`: Agglomerative的距离阈值（默认0.5）
         - `linkage`: Agglomerative的链接方式（默认"average"，支持cosine距离）
         - `metric`: 距离度量（默认"cosine"，适用于已归一化的向量）

9. **ClusterInsightAgent**: 簇命名、摘要、建议（LLM生成）
   - 输入：cluster_stats, aspect_sentiment_valid, review_sentences
   - 输出：cluster_reports
   - 功能：
     - 使用LLM为每个簇生成洞察报告（簇名称、摘要、优先级、证据、建议）
     - 支持结构化输出（Pydantic模型约束）
     - 自动验证证据可回溯性（确保evidence_items中的sentence_id来自输入样本）
     - LLM失败时自动降级到简化版本（fallback机制）
     - 带重试机制（指数退避）
   - 配置参数：
     - `llm_wrapper`: LLMWrapper实例（从配置中传入，使用`models.insight_llm`配置）
     - `prompt_version`: Prompt版本（默认v1.0，从configs/prompts/insight/加载）
     - `max_retries`: 最大重试次数（默认3）
   - 模型配置：
     - 在`configs/runs/*.yaml`中通过`models.insight_llm`配置
     - 推荐使用更大的模型（如`qwen3:14b`），temperature=0.3，更有创造性
     - 如果未指定`insight_llm`，则使用`models.llm_provider`和`models.llm_model`作为默认值
   - 注意：
     - Prompt模板在`configs/prompts/insight/v1.0.md`中定义
     - 输出结构符合附录A规范（ClusterInsightAgent输出规范）
     - 与`AspectSentimentExtractorAgent`可以使用不同的模型，以满足不同任务的需求

10. **ReportAssemblerAgent**: 报告组装
    - 输入：cluster_reports, cluster_stats, evaluation_metrics, 各中间表（用于统计）
    - 输出：final_report.md
    - 功能：
      - 生成包含聚类效果参数的完整报告
      - 包含数据统计、评估指标、簇详细分析
      - 展示簇大小、负面率、置信度等聚类效果参数
      - **聚类质量指标**：
        - LLM置信度：LLM生成的洞察置信度
        - 聚类置信度：基于簇内一致性和分离度计算的置信度
        - 簇内平均距离：簇内样本到簇中心的平均距离
        - 簇间最小距离：到最近簇的距离
        - 分离度比率：簇间距离/簇内距离（越大越好）
        - 紧密度：簇内距离的倒数（越大越好）
        - 情感一致性：簇内情感分布的均匀程度
      - 自动解析JSON字段（evidence_items, action_items）并格式化显示
      - 包含空值处理和异常处理，确保aspect_norm等关键字段正确显示
      - 添加调试日志，便于排查数据问题

11. **EvaluationAgent**: 自动指标 + 人工抽样包
    - 输入：各中间表
    - 输出：evaluation_metrics
    - 功能：
      - 计算覆盖率指标（候选覆盖率、抽取成功率、有效aspect率、有效句子率）
      - 诊断数据重复问题：
        - **真正的重复检测**：检查相同的(sentence_id, aspect_norm, issue_norm)组合，这是真正的重复
        - **组合重复检测**：检查(sentence_id, aspect_norm)组合，区分两种情况：
          - 正常情况：有多个不同的issue_norm（一个句子可以针对同一个aspect提出多个问题）
          - 异常情况：有相同的issue_norm（可能是真正的重复）
      - 自动检测异常指标（如有效aspect率>100%）

## 基类

所有Agent继承自`BaseAgent`，必须实现`process()`方法。

## 日志输出

所有Agent都集成了步骤相关的日志输出功能，使用`src.utils.logger.get_logger()`获取日志记录器。

日志输出包括：
- **开始执行**：记录Agent开始执行，包含run_id和关键参数
- **处理进度**：对于大批量处理，每处理一定数量记录后输出进度（如每100条、每50条）
- **关键步骤**：记录重要的处理步骤，如向量化、聚类、统计计算等
- **完成状态**：记录处理完成，包含统计信息（处理数量、成功/失败数量等）

日志级别：
- `INFO`：正常流程信息（开始、进度、完成）
- `WARNING`：警告信息（如未找到数据、跳过记录等）
- `ERROR`：错误信息（处理失败、异常等）
- `DEBUG`：调试信息（详细的计算过程，仅在需要时使用）

示例日志输出：
```
INFO - 开始执行DataSelectorAgent，run_id=20260103-1530_default
INFO - 执行数据查询，参数: ['Appliances']
INFO - 查询结果数量: 1500
INFO - 已插入 100 条记录...
INFO - 已插入 200 条记录...
INFO - DataSelectorAgent完成: 共插入 1500 条记录
```

