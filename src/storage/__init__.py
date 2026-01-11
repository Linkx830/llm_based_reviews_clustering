# 新增或修改我时需要修改这个文件夹中的README.md文件
"""DuckDB存储管理模块"""
from .connection import DuckDBConnection
from .table_manager import TableManager
from .version_fields import VersionFields

__all__ = ["DuckDBConnection", "TableManager", "VersionFields"]

