# 新增或修改我时需要修改这个文件夹中的README.md文件
"""DuckDB查询工具"""
from typing import List, Dict, Any, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from ..storage.connection import DuckDBConnection


class DuckDBQueryInput(BaseModel):
    """DuckDB查询输入"""
    query: str = Field(description="SQL查询语句")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="查询参数")


class DuckDBQueryTool(BaseTool):
    """
    DuckDB查询工具
    
    职责：
    - 封装DuckDB读写操作
    - 批处理支持
    - 事务管理
    """
    name: str = "duckdb_query"
    description: str = "执行DuckDB查询，支持读写操作"
    args_schema: Type[BaseModel] = DuckDBQueryInput
    
    def __init__(self, db_conn: DuckDBConnection, read_only: bool = False):
        super().__init__()
        self.db = db_conn
        self.read_only = read_only
    
    def _run(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List:
        """执行查询"""
        if self.read_only:
            return self.db.execute_read(query, parameters)
        else:
            # 判断是读还是写
            query_lower = query.strip().lower()
            if query_lower.startswith(("select", "show", "describe", "explain")):
                return self.db.execute_read(query, parameters)
            else:
                self.db.execute_write(query, parameters)
                return []
    
    async def _arun(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List:
        """异步执行查询"""
        return self._run(query, parameters)

