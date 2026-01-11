# 新增或修改我时需要修改这个文件夹中的README.md文件
"""
清理指定run_id在clustering及之后步骤产生的数据

用于从clustering步骤恢复运行前，清理之前可能存在的部分结果数据
"""
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.connection import DuckDBConnection
from src.storage.table_manager import TableManager
from src.utils.logger import setup_logger
import logging


def cleanup_after_step(run_id: str, step_name: str, db_path: str = None):
    """
    清理指定run_id在指定步骤及之后步骤产生的数据
    
    Args:
        run_id: 运行ID
        step_name: 步骤名称（如 'clustering', 'insight', 'evaluation', 'report'）
        db_path: DuckDB文件路径
    """
    if db_path is None:
        db_path = os.getenv("DUCKDB_PATH", "data/duckDB/amazon.duckdb")
    
    # 设置日志
    setup_logger("root", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info(f"开始清理 run_id={run_id} 在步骤 '{step_name}' 及之后的数据")
    logger.info("=" * 60)
    
    # 连接数据库
    db_conn = DuckDBConnection.get_instance(db_path)
    db_conn.connect()
    table_manager = TableManager(db_conn)
    
    # 定义步骤到表的映射
    step_to_tables = {
        "clustering": [
            TableManager.ISSUE_CLUSTERS,
            TableManager.CLUSTER_STATS,
            TableManager.CLUSTER_REPORTS,  # insight步骤产生的
            TableManager.EVALUATION_METRICS,  # evaluation步骤产生的
        ],
        "insight": [
            TableManager.CLUSTER_REPORTS,
            TableManager.EVALUATION_METRICS,
        ],
        "evaluation": [
            TableManager.EVALUATION_METRICS,
        ],
        "report": []  # report步骤不写数据库，只写文件
    }
    
    # 获取需要清理的表（包括指定步骤及之后的所有表）
    steps_order = ["clustering", "insight", "evaluation", "report"]
    if step_name not in steps_order:
        logger.error(f"未知的步骤名称: {step_name}")
        logger.error(f"支持的步骤: {', '.join(steps_order)}")
        return
    
    step_idx = steps_order.index(step_name)
    tables_to_clean = []
    for i in range(step_idx, len(steps_order)):
        step = steps_order[i]
        tables_to_clean.extend(step_to_tables.get(step, []))
    
    # 去重
    tables_to_clean = list(set(tables_to_clean))
    
    if not tables_to_clean:
        logger.info(f"步骤 '{step_name}' 及之后没有需要清理的数据库表")
    else:
        logger.info(f"将清理以下表: {', '.join(tables_to_clean)}")
        
        # 清理每个表
        total_deleted = 0
        for table_name in tables_to_clean:
            # 检查表是否存在
            if not table_manager.table_exists(table_name):
                logger.warning(f"表 {table_name} 不存在，跳过")
                continue
            
            # 先查询要删除的记录数
            count_query = f"SELECT COUNT(*) FROM {table_name} WHERE run_id = ?"
            try:
                count_result = db_conn.execute_read(count_query, [run_id])
                deleted_count = count_result[0][0] if count_result else 0
                
                if deleted_count > 0:
                    # 删除数据
                    delete_query = f"DELETE FROM {table_name} WHERE run_id = ?"
                    db_conn.execute_write(delete_query, [run_id])
                    total_deleted += deleted_count
                    logger.info(f"  表 {table_name}: 删除了 {deleted_count} 条记录")
                else:
                    logger.info(f"  表 {table_name}: 没有需要删除的记录")
            except Exception as e:
                logger.error(f"  表 {table_name}: 删除失败 - {str(e)}")
        
        logger.info(f"总共删除了 {total_deleted} 条记录")
    
    # 清理run_log中相关步骤的记录（可选）
    if table_manager.table_exists(TableManager.RUN_LOG):
        step_names = steps_order[step_idx:]
        if step_names:
            placeholders = ','.join(['?' for _ in step_names])
            delete_log_query = f"""
                DELETE FROM {TableManager.RUN_LOG} 
                WHERE run_id = ? AND step_name IN ({placeholders})
            """
            try:
                # 先查询数量
                count_log_query = f"""
                    SELECT COUNT(*) FROM {TableManager.RUN_LOG} 
                    WHERE run_id = ? AND step_name IN ({placeholders})
                """
                params = [run_id] + step_names
                count_result = db_conn.execute_read(count_log_query, params)
                deleted_logs = count_result[0][0] if count_result else 0
                
                if deleted_logs > 0:
                    db_conn.execute_write(delete_log_query, params)
                    logger.info(f"删除了 {deleted_logs} 条运行日志记录")
            except Exception as e:
                logger.warning(f"清理运行日志失败: {str(e)}")
    
    logger.info("=" * 60)
    logger.info("清理完成！")
    logger.info("=" * 60)
    
    db_conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="清理指定run_id在指定步骤及之后步骤产生的数据"
    )
    parser.add_argument("--run-id", type=str, required=True, help="运行ID")
    parser.add_argument(
        "--step", 
        type=str, 
        required=True,
        choices=["clustering", "insight", "evaluation", "report"],
        help="从哪个步骤开始清理（会清理该步骤及之后的所有数据）"
    )
    parser.add_argument(
        "--db-path", 
        type=str, 
        default=None,
        help="DuckDB文件路径（默认从环境变量DUCKDB_PATH或data/duckDB/amazon.duckdb）"
    )
    
    args = parser.parse_args()
    
    cleanup_after_step(args.run_id, args.step, args.db_path)


if __name__ == "__main__":
    main()

