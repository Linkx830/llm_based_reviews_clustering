# 新增或修改我时需要修改这个文件夹中的README.md文件
"""
导出簇信息和簇中的每条评论到CSV文件

接受一个run_id，读取簇信息以及簇中的每一条评论，并输出为CSV文件到所在run文件夹中
"""
import argparse
import os
import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.connection import DuckDBConnection
from src.storage.table_manager import TableManager
from src.utils.logger import setup_logger
import logging

logger = logging.getLogger(__name__)


def export_clusters_to_csv(run_id: str, db_path: str = None, output_dir: Path = None):
    """
    导出簇信息和簇中的每条评论到CSV文件
    
    Args:
        run_id: 运行ID
        db_path: DuckDB文件路径
        output_dir: 输出目录（如果为None，则使用outputs/runs/<run_id>/）
    """
    if db_path is None:
        db_path = os.getenv("DUCKDB_PATH", "data/duckDB/amazon.duckdb")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = project_root / "outputs" / "runs" / run_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    setup_logger("root", level=logging.INFO)
    logger.info("=" * 60)
    logger.info(f"开始导出 run_id={run_id} 的簇信息和评论")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    # 连接数据库
    db_conn = DuckDBConnection.get_instance(db_path)
    db_conn.connect()
    table_manager = TableManager(db_conn)
    
    # 检查必要的表是否存在
    required_tables = [
        TableManager.ISSUE_CLUSTERS,
        TableManager.CLUSTER_STATS,
        TableManager.REVIEW_SENTENCES
    ]
    
    for table_name in required_tables:
        if not table_manager.table_exists(table_name):
            logger.error(f"表 {table_name} 不存在，无法导出")
            db_conn.close()
            return
    
    # 查询簇信息和评论
    # 使用LEFT JOIN将簇信息、簇统计、评论句子、簇洞察信息合并
    query = f"""
        SELECT 
            -- 簇基本信息
            ic.aspect_norm,
            ic.cluster_id,
            ic.is_noise,
            ic.issue_norm,
            ic.sentiment,
            ic.cluster_key_text,
            
            -- 簇统计信息
            cs.cluster_size,
            cs.neg_ratio,
            cs.intra_cluster_distance,
            cs.inter_cluster_distance,
            cs.separation_ratio,
            cs.cohesion,
            cs.cluster_confidence,
            cs.sentiment_consistency,
            cs.representative_sentence_ids,
            
            -- 簇洞察信息（如果有）
            cr.cluster_name,
            cr.summary,
            cr.priority,
            cr.confidence as insight_confidence,
            
            -- 评论句子信息
            rs.sentence_id,
            rs.review_pk,
            rs.parent_asin,
            rs.timestamp,
            rs.rating,
            rs.verified_purchase,
            rs.helpful_vote,
            rs.sentence_index,
            rs.target_sentence,
            rs.prev_sentence,
            rs.next_sentence,
            rs.context_text
            
        FROM {TableManager.ISSUE_CLUSTERS} ic
        LEFT JOIN {TableManager.CLUSTER_STATS} cs
            ON ic.run_id = cs.run_id 
            AND ic.aspect_norm = cs.aspect_norm 
            AND ic.cluster_id = cs.cluster_id
        LEFT JOIN {TableManager.CLUSTER_REPORTS} cr
            ON ic.run_id = cr.run_id 
            AND ic.aspect_norm = cr.aspect_norm 
            AND ic.cluster_id = cr.cluster_id
        LEFT JOIN {TableManager.REVIEW_SENTENCES} rs
            ON ic.run_id = rs.run_id 
            AND ic.sentence_id = rs.sentence_id
        WHERE ic.run_id = ?
        ORDER BY 
            ic.aspect_norm,
            CASE WHEN ic.cluster_id = 'noise' THEN 1 ELSE 0 END,
            CASE 
                WHEN ic.cluster_id = 'noise' THEN 'zzz_noise'
                ELSE LPAD(ic.cluster_id, 10, '0')
            END,
            rs.timestamp DESC
    """
    
    logger.info("正在查询数据...")
    results = db_conn.execute_read(query, [run_id])
    
    if not results:
        logger.warning(f"run_id={run_id} 没有找到任何簇数据")
        db_conn.close()
        return
    
    logger.info(f"查询到 {len(results)} 条记录")
    
    # 准备CSV数据
    csv_rows = []
    for row in results:
        csv_row = {
            # 簇基本信息
            "aspect_norm": row[0] or "",
            "cluster_id": row[1] or "",
            "is_noise": "是" if row[2] else "否",
            "issue_norm": row[3] or "",
            "sentiment": row[4] or "",
            "cluster_key_text": row[5] or "",
            
            # 簇统计信息
            "cluster_size": row[6] if row[6] is not None else "",
            "neg_ratio": f"{row[7]:.4f}" if row[7] is not None else "",
            "intra_cluster_distance": f"{row[8]:.4f}" if row[8] is not None else "",
            "inter_cluster_distance": f"{row[9]:.4f}" if row[9] is not None else "",
            "separation_ratio": f"{row[10]:.4f}" if row[10] is not None else "",
            "cohesion": f"{row[11]:.4f}" if row[11] is not None else "",
            "cluster_confidence": f"{row[12]:.4f}" if row[12] is not None else "",
            "sentiment_consistency": f"{row[13]:.4f}" if row[13] is not None else "",
            "representative_sentence_ids": row[14] or "",
            
            # 簇洞察信息
            "cluster_name": row[15] or "",
            "summary": row[16] or "",
            "priority": row[17] or "",
            "insight_confidence": f"{row[18]:.4f}" if row[18] is not None else "",
            
            # 评论句子信息
            "sentence_id": row[19] or "",
            "review_pk": row[20] or "",
            "parent_asin": row[21] or "",
            "timestamp": row[22] if row[22] is not None else "",
            "rating": f"{row[23]:.1f}" if row[23] is not None else "",
            "verified_purchase": "是" if row[24] else "否" if row[24] is not None else "",
            "helpful_vote": row[25] if row[25] is not None else "",
            "sentence_index": row[26] if row[26] is not None else "",
            "target_sentence": (row[27] or "").replace("\n", " ").replace("\r", " "),
            "prev_sentence": (row[28] or "").replace("\n", " ").replace("\r", " ") if row[28] else "",
            "next_sentence": (row[29] or "").replace("\n", " ").replace("\r", " ") if row[29] else "",
            "context_text": (row[30] or "").replace("\n", " ").replace("\r", " ") if row[30] else "",
        }
        csv_rows.append(csv_row)
    
    # 定义CSV列顺序
    csv_columns = [
        # 簇基本信息
        "aspect_norm",
        "cluster_id",
        "cluster_name",
        "is_noise",
        "issue_norm",
        "sentiment",
        "cluster_key_text",
        "priority",
        
        # 簇统计信息
        "cluster_size",
        "neg_ratio",
        "intra_cluster_distance",
        "inter_cluster_distance",
        "separation_ratio",
        "cohesion",
        "cluster_confidence",
        "sentiment_consistency",
        "insight_confidence",
        "representative_sentence_ids",
        
        # 簇洞察信息
        "summary",
        
        # 评论句子信息
        "sentence_id",
        "review_pk",
        "parent_asin",
        "timestamp",
        "rating",
        "verified_purchase",
        "helpful_vote",
        "sentence_index",
        "target_sentence",
        "prev_sentence",
        "next_sentence",
        "context_text",
    ]
    
    # 写入CSV文件
    csv_file = output_dir / "clusters_and_reviews.csv"
    logger.info(f"正在写入CSV文件: {csv_file}")
    
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_rows)
    
    logger.info(f"成功导出 {len(csv_rows)} 条记录到 {csv_file}")
    
    # 生成统计摘要
    summary = {
        "run_id": run_id,
        "total_records": len(csv_rows),
        "total_clusters": len(set(row["cluster_id"] for row in csv_rows if row["cluster_id"] != "noise")),
        "noise_records": sum(1 for row in csv_rows if row["is_noise"] == "是"),
        "aspects": sorted(set(row["aspect_norm"] for row in csv_rows if row["aspect_norm"])),
    }
    
    summary_file = output_dir / "clusters_export_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"统计摘要已保存到: {summary_file}")
    logger.info(f"  - 总记录数: {summary['total_records']}")
    logger.info(f"  - 总簇数: {summary['total_clusters']}")
    logger.info(f"  - 噪声记录数: {summary['noise_records']}")
    logger.info(f"  - Aspect数: {len(summary['aspects'])}")
    
    logger.info("=" * 60)
    logger.info("导出完成！")
    logger.info("=" * 60)
    
    db_conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="导出簇信息和簇中的每条评论到CSV文件"
    )
    parser.add_argument("--run-id", type=str, required=True, help="运行ID")
    parser.add_argument(
        "--db-path", 
        type=str, 
        default=None,
        help="DuckDB文件路径（默认从环境变量DUCKDB_PATH或data/duckDB/amazon.duckdb）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认使用outputs/runs/<run_id>/）"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    export_clusters_to_csv(args.run_id, args.db_path, output_dir)


if __name__ == "__main__":
    main()

