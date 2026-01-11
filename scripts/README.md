# Scripts 目录说明

本目录包含用于维护和管理项目的实用脚本。

## 文件说明

### cleanup_resume.py

清理指定run_id在指定步骤及之后步骤产生的数据，用于从某个步骤恢复运行前清理可能存在的部分结果数据。

**使用场景：**
- 当流水线在某个步骤（如clustering）中途停止时
- 需要从该步骤重新运行，但之前可能已经产生了部分数据
- 为了避免数据重复，需要先清理该步骤及之后的数据

**使用方法：**

```bash
# 清理clustering及之后的所有数据（包括insight、evaluation、report步骤产生的数据）
python scripts/cleanup_resume.py --run-id <your_run_id> --step clustering

# 清理insight及之后的数据
python scripts/cleanup_resume.py --run-id <your_run_id> --step insight

# 指定数据库路径
python scripts/cleanup_resume.py --run-id <your_run_id> --step clustering --db-path data/duckDB/amazon.duckdb
```

**清理的表：**
- `clustering` 步骤：清理 `issue_clusters`, `cluster_stats`, `cluster_reports`, `evaluation_metrics`
- `insight` 步骤：清理 `cluster_reports`, `evaluation_metrics`
- `evaluation` 步骤：清理 `evaluation_metrics`
- `report` 步骤：无数据库表需要清理（只写文件）

**注意事项：**
- 此脚本只清理数据库中的数据，不会删除输出文件（如 `outputs/runs/<run_id>/` 目录下的文件）
- 如果需要清理输出文件，请手动删除对应的目录
- 清理前请确认run_id正确，避免误删其他运行的数据

---

### export_clusters_to_csv.py

导出指定run_id的簇信息和簇中的每条评论到CSV文件。

**使用场景：**
- 需要将簇信息和评论数据导出为CSV格式进行分析
- 需要查看某个run_id的所有簇及其包含的评论详情
- 需要将数据导出到Excel或其他工具进行进一步处理

**使用方法：**

```bash
# 导出指定run_id的簇信息和评论
python scripts/export_clusters_to_csv.py --run-id <your_run_id>

# 指定数据库路径
python scripts/export_clusters_to_csv.py --run-id <your_run_id> --db-path data/duckDB/amazon.duckdb

# 指定输出目录（默认使用outputs/runs/<run_id>/）
python scripts/export_clusters_to_csv.py --run-id <your_run_id> --output-dir /path/to/output
```

**输出文件：**
- `outputs/runs/<run_id>/clusters_and_reviews.csv`: 包含所有簇信息和评论的CSV文件
- `outputs/runs/<run_id>/clusters_export_summary.json`: 导出统计摘要（总记录数、簇数、噪声记录数等）

**CSV文件包含的字段：**
- 簇基本信息：aspect_norm, cluster_id, cluster_name, is_noise, issue_norm, sentiment, cluster_key_text, priority
- 簇统计信息：cluster_size, neg_ratio, intra_cluster_distance, inter_cluster_distance, separation_ratio, cohesion, cluster_confidence, sentiment_consistency, insight_confidence, representative_sentence_ids
- 簇洞察信息：summary
- 评论句子信息：sentence_id, review_pk, parent_asin, timestamp, rating, verified_purchase, helpful_vote, sentence_index, target_sentence, prev_sentence, next_sentence, context_text

**注意事项：**
- 导出的CSV文件使用UTF-8-BOM编码，可以在Excel中正确显示中文
- 如果某个簇没有对应的cluster_reports记录，相关字段（如cluster_name, summary）将为空
- 如果某个记录没有对应的review_sentences记录，相关字段将为空

