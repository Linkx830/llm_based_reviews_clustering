# 新增或修改我时需要修改这个文件夹中的README.md文件
"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..storage.connection import DuckDBConnection
from ..storage.version_fields import VersionFields


class BaseAgent(ABC):
    """
    Agent基类
    
    所有Agent必须继承此类并实现process方法
    """
    
    def __init__(
        self,
        db_conn: DuckDBConnection,
        run_id: str,
        pipeline_version: str,
        data_slice_id: str
    ):
        """
        Args:
            db_conn: DuckDB连接
            run_id: 运行ID
            pipeline_version: 管道版本
            data_slice_id: 数据切片ID
        """
        self.db = db_conn
        self.run_id = run_id
        self.pipeline_version = pipeline_version
        self.data_slice_id = data_slice_id
        self.version_fields = VersionFields.get_base_version_fields(
            run_id, pipeline_version, data_slice_id
        )
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        处理逻辑（子类必须实现）
        
        Returns:
            处理结果字典
        """
        pass
    
    def get_version_fields(self) -> Dict[str, Any]:
        """获取版本字段"""
        return self.version_fields.copy()

