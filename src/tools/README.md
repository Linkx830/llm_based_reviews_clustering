# tools/ 模块说明

## 职责

LangChain Tools封装模块。

## 工具列表

- `duckdb_tool.py`: DuckDBQueryTool - DuckDB查询工具（读写/批处理/事务）
- `structured_output_tool.py`: StructuredOutputTool - 结构化输出解析/校验/重试
- `embedding_tool.py`: EmbeddingTool - 向量化服务抽象（支持MRL维度裁剪）
- `clustering_tool.py`: ClusteringTool - 聚类执行与簇统计
- `reranker_tool.py`: RerankerTool - Reranker二次验证工具（用于阶段D边界精修）
- `logging_tool.py`: LoggingTool - run_log写入、指标快照

## 使用示例

```python
from src.tools import EmbeddingTool, ClusteringTool

# Embedding - sentence-transformers（本地模型，支持批处理）
embedding = EmbeddingTool("all-MiniLM-L6-v2", batch_size=32)
vectors = embedding.encode(["text1", "text2"])

# Embedding - Ollama（远程API，支持并发请求加速）
embedding_ollama = EmbeddingTool(
    "qwen3-embedding:0.6b",
    base_url="http://localhost:11434",
    max_workers=4,  # 并发请求数，可根据服务器性能调整（建议2-8）
    batch_size=32   # Ollama时作为并发数使用
)
vectors = embedding_ollama.encode(["text1", "text2"])

# Embedding - 支持instruction-aware（用于任务特定的embedding）
instruction = "Represent the underlying customer issue type for clustering."
vectors_with_instruction = embedding.encode(
    ["text1", "text2"],
    instruction=instruction,  # 添加instruction
    normalize=True  # L2归一化（默认True，推荐用于cosine相似度）
)

# Embedding - 支持MRL维度裁剪（仅支持支持MRL的模型如Qwen3-Embedding）
embedding_mrl = EmbeddingTool(
    "qwen3-embedding:0.6b",
    base_url="http://localhost:11434",
    mrl_dimensions=512  # 裁剪到512维（可选：768/512/256）
)
vectors_mrl = embedding_mrl.encode(["text1", "text2"])

# Reranker - 用于阶段D的边界精修
from src.tools import RerankerTool

reranker = RerankerTool(
    model_name="qwen3-reranker:0.6b",  # Ollama reranker模型
    base_url="http://localhost:11434",
    score_threshold=0.6  # 相关性分数阈值
)
scores = reranker.rerank_pairs(
    query_texts=["Aspect: battery\nIssue: drains fast"],
    document_texts=["Aspect: battery\nIssue: runs out quickly"],
    max_workers=4
)

# Clustering - HDBSCAN（适合大数据集，自动确定簇数，支持cosine距离）
clusterer_hdbscan = ClusteringTool(
    method="hdbscan",
    metric="cosine",  # 使用cosine距离（适用于已归一化的向量）
    # 注意：HDBSCAN内部会将'cosine'自动转换为'euclidean'（向量已归一化时等价）
    min_cluster_size=5,
    min_samples=3
)
labels = clusterer_hdbscan.fit(vectors)  # vectors应该是已归一化的

# Clustering - Agglomerative（适合小数据集，自动确定簇数，支持cosine距离）
clusterer_agg = ClusteringTool(
    method="agglomerative",
    metric="cosine",  # 使用cosine距离
    distance_threshold=0.5,  # 基于距离阈值自动确定簇数
    linkage="average"  # 使用average而不是ward（ward只支持euclidean）
)
labels = clusterer_agg.fit(vectors)
```

## EmbeddingTool 性能优化

### 批处理与并发
- **sentence-transformers**：使用批处理（`batch_size`）加速，默认32
- **Ollama API**：使用多线程并发请求（`max_workers`）加速，默认4个并发
  - 原来逐个处理182个文本需要约6-9分钟
  - 使用4个并发后，预计可缩短到约2-3分钟（取决于服务器性能）
  - 可根据服务器性能调整`max_workers`（建议2-8）

### MRL维度裁剪
- **Qwen3-Embedding**支持MRL（Matryoshka Representation Learning）
- 可以通过`mrl_dimensions`参数裁剪向量维度（如768/512/256）
- 降低维度可以：
  - 减少存储空间
  - 加速kNN检索
  - 提高推理速度
- 注意：不同维度可能影响聚类质量，建议做小规模ablation实验

### 配置建议
- 本地GPU：`max_workers=8`（如果Ollama支持）
- 本地CPU：`max_workers=4`
- 远程服务器：`max_workers=2-4`（避免过载）

## ClusteringTool 注意事项

### HDBSCAN 指标转换
- **'cosine' 指标自动转换**：由于 HDBSCAN 内部使用的 sklearn BallTree 不支持 'cosine' 字符串指标，当指定 `metric="cosine"` 时，会自动转换为 `"euclidean"`
- **数学等价性**：对于已归一化的向量（L2 normalized），euclidean 距离与 cosine 距离在聚类结果上等价
  - euclidean² = 2 × (1 - cosine_similarity) = 2 × cosine_distance
  - 距离的相对顺序保持不变，因此聚类结果相同
- **其他方法**：AgglomerativeClustering 和 DBSCAN 直接支持 'cosine' 指标，无需转换

## RerankerTool 使用说明

### 功能
- 对embedding近邻的候选对进行reranker打分
- 用于阶段D的边界精修，减少误聚/漏聚
- 支持Ollama reranker模型和LLM-based reranker
- **自动回退机制**：当Ollama reranker API不可用时，自动回退到LLM-based reranker（如果提供了LLM wrapper）

### 使用场景
- 聚类边界样本的二次验证
- 提高聚类质量，减少噪声
- 成本可控（只处理embedding近邻，不处理全量两两对）

### 错误处理
- 如果Ollama reranker API返回404（端点不存在），会自动回退到LLM-based reranker
- 如果未提供LLM wrapper且Ollama reranker失败，会记录错误并返回默认低分（0.0）
- 建议：在配置中启用reranker时，系统会自动提供extraction_llm作为回退选项

