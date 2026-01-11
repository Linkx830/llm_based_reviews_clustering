# 新增或修改我时需要修改这个文件夹中的README.md文件
"""版本字段管理模块"""
from datetime import datetime
from typing import Dict, Any
import hashlib


class VersionFields:
    """
    版本字段生成与管理
    
    职责：
    - 生成run_id、data_slice_id等版本标识
    - 提供版本字段的标准格式
    """
    
    @staticmethod
    def generate_run_id(scope: str = "default", tag: str = "") -> str:
        """
        生成run_id
        
        格式：YYYYMMDD-HHMM_<scope>_<shorttag>
        例：20260103-1530_category_headphones_v11
        """
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M")
        parts = [timestamp, scope]
        if tag:
            parts.append(tag)
        return "_".join(parts)
    
    @staticmethod
    def generate_data_slice_id(
        main_category: str = None,
        parent_asin: str = None,
        time_window: tuple = None,
        filters: Dict[str, Any] = None
    ) -> str:
        """
        生成data_slice_id（数据切片标识）
        
        基于切片条件生成确定性哈希
        """
        parts = []
        if main_category:
            parts.append(f"category:{main_category}")
        if parent_asin:
            parts.append(f"asin:{parent_asin}")
        if time_window:
            parts.append(f"time:{time_window[0]}-{time_window[1]}")
        if filters:
            filter_str = ",".join(f"{k}={v}" for k, v in sorted(filters.items()))
            parts.append(f"filters:{filter_str}")
        
        if not parts:
            return "default"
        
        content = "|".join(parts)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @staticmethod
    def get_base_version_fields(
        run_id: str,
        pipeline_version: str,
        data_slice_id: str,
        source_snapshot_at: datetime = None
    ) -> Dict[str, Any]:
        """
        获取基础版本字段字典
        
        Returns:
            包含run_id, pipeline_version, data_slice_id, created_at, source_snapshot_at的字典
        """
        fields = {
            "run_id": run_id,
            "pipeline_version": pipeline_version,
            "data_slice_id": data_slice_id,
            "created_at": datetime.now(),
        }
        if source_snapshot_at:
            fields["source_snapshot_at"] = source_snapshot_at
        return fields
    
    @staticmethod
    def get_model_version_fields(
        llm_model: str = None,
        prompt_version: str = None,
        embedding_model: str = None,
        clustering_config_id: str = None
    ) -> Dict[str, Any]:
        """
        获取模型与Prompt版本字段
        
        Returns:
            包含模型相关版本字段的字典
        """
        fields = {}
        if llm_model:
            fields["llm_model"] = llm_model
        if prompt_version:
            fields["prompt_version"] = prompt_version
        if embedding_model:
            fields["embedding_model"] = embedding_model
        if clustering_config_id:
            fields["clustering_config_id"] = clustering_config_id
        return fields

