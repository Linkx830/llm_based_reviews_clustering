# 新增或修改我时需要修改这个文件夹中的README.md文件
"""运行日志工具"""
from datetime import datetime
from typing import Optional
from ..storage.connection import DuckDBConnection
from ..storage.table_manager import TableManager


class LoggingTool:
    """
    运行日志工具
    
    职责：
    - 记录运行日志到run_log表
    - 指标快照
    """
    
    def __init__(self, db_conn: DuckDBConnection):
        self.db = db_conn
        self.table_manager = TableManager(db_conn)
    
    def log_step_start(
        self,
        run_id: str,
        step_name: str,
        input_rows: Optional[int] = None
    ):
        """记录步骤开始"""
        query = """
            INSERT INTO run_log 
            (run_id, step_name, status, input_rows, started_at)
            VALUES (?, ?, 'STARTED', ?, ?)
        """
        # 使用位置参数
        self.db.execute_write(
            query,
            [run_id, step_name, input_rows, datetime.now()]
        )
    
    def log_step_success(
        self,
        run_id: str,
        step_name: str,
        output_rows: Optional[int] = None,
        message: Optional[str] = None
    ):
        """记录步骤成功"""
        query = """
            UPDATE run_log
            SET status = 'SUCCESS',
                output_rows = ?,
                finished_at = ?,
                message = ?
            WHERE run_id = ? AND step_name = ?
        """
        # 使用位置参数
        self.db.execute_write(
            query,
            [output_rows, datetime.now(), message, run_id, step_name]
        )
    
    def log_step_failed(
        self,
        run_id: str,
        step_name: str,
        error_rows: Optional[int] = None,
        message: Optional[str] = None
    ):
        """记录步骤失败"""
        query = """
            UPDATE run_log
            SET status = 'FAILED',
                error_rows = ?,
                finished_at = ?,
                message = ?
            WHERE run_id = ? AND step_name = ?
        """
        # 使用位置参数
        self.db.execute_write(
            query,
            [error_rows, datetime.now(), message, run_id, step_name]
        )

