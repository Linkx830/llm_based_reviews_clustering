# 新增或修改我时需要修改这个文件夹中的README.md文件
"""DuckDB连接管理模块 - 单写者模式"""
import duckdb
from typing import Optional
import threading
from pathlib import Path


class DuckDBConnection:
    """
    DuckDB连接管理器 - 单写者模式
    
    职责：
    - 管理DuckDB连接（单例模式，确保单写者）
    - 提供读写接口
    - 事务管理
    """
    
    _instance: Optional['DuckDBConnection'] = None
    _lock = threading.Lock()
    
    def __init__(self, db_path: str):
        if DuckDBConnection._instance is not None:
            raise RuntimeError("DuckDBConnection is singleton, use get_instance()")
        self.db_path = Path(db_path)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._write_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> 'DuckDBConnection':
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if db_path is None:
                        raise ValueError("db_path required for first initialization")
                    cls._instance = cls(db_path)
        return cls._instance
    
    def connect(self):
        """建立连接"""
        if self.conn is None:
            self.conn = duckdb.connect(str(self.db_path))
    
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def execute_read(self, query: str, parameters: Optional[list] = None):
        """
        执行只读查询
        
        Args:
            query: SQL查询语句（使用?占位符）
            parameters: 查询参数（列表或元组，位置参数）
                       DuckDB使用?占位符时需要位置参数
        
        Returns:
            查询结果
        """
        if self.conn is None:
            self.connect()
        if parameters:
            # DuckDB使用位置参数（列表或元组）
            if isinstance(parameters, dict):
                # 如果是字典，尝试按字典值的顺序转换为列表
                # 注意：这要求字典保持插入顺序（Python 3.7+）
                parameters = list(parameters.values())
            return self.conn.execute(query, parameters).fetchall()
        return self.conn.execute(query).fetchall()
    
    def execute_write(self, query: str, parameters: Optional[list] = None):
        """
        执行写入操作（串行化，单写者模式）
        
        Args:
            query: SQL语句
            parameters: 查询参数（列表或元组，位置参数）
                       DuckDB使用?占位符时需要位置参数
        """
        with self._write_lock:
            if self.conn is None:
                self.connect()
            if parameters:
                # DuckDB使用位置参数（列表或元组）
                if isinstance(parameters, dict):
                    # 如果是字典，尝试按字典值的顺序转换为列表
                    # 注意：这要求字典保持插入顺序（Python 3.7+）
                    parameters = list(parameters.values())
                self.conn.execute(query, parameters)
            else:
                self.conn.execute(query)
            self.conn.commit()
    
    def execute_many(self, query: str, data: list):
        """
        批量执行写入
        
        Args:
            query: SQL语句
            data: 数据列表
        """
        with self._write_lock:
            if self.conn is None:
                self.connect()
            self.conn.executemany(query, data)
            self.conn.commit()
    
    def create_table_if_not_exists(self, table_name: str, schema: str):
        """
        创建表（如果不存在）
        
        Args:
            table_name: 表名
            schema: 表结构SQL
        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.execute_write(query)

