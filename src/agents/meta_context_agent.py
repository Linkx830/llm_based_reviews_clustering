# 新增或修改我时需要修改这个文件夹中的README.md文件
"""MetaContextAgent - 元数据上下文构建（截断/摘要）"""
from typing import Dict, Any
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetaContextAgent(BaseAgent):
    """
    MetaContextAgent
    
    职责：
    - 为每个parent_asin构建"LLM可用的最小上下文"
    - 对features/description/details执行截断或摘要
    
    输入表：meta, selected_reviews
    输出表：meta_context
    """
    
    def __init__(self, *args, context_version: str = "truncate_512", **kwargs):
        super().__init__(*args, **kwargs)
        self.context_version = context_version
        self.max_length = 2048  # 默认截断长度
    
    def process(self) -> Dict[str, Any]:
        """
        构建元数据上下文
        
        Returns:
            处理结果统计
        """
        logger.info(f"开始执行MetaContextAgent，run_id={self.run_id}, context_version={self.context_version}")
        
        # 获取所有唯一的parent_asin
        query = f"""
            SELECT DISTINCT parent_asin
            FROM {TableManager.SELECTED_REVIEWS}
            WHERE run_id = ?
        """
        parent_asins = self.db.execute_read(query, {"run_id": self.run_id})
        logger.info(f"找到 {len(parent_asins)} 个唯一的 parent_asin")
        
        # 获取meta信息
        table_manager = TableManager(self.db)
        inserted_count = 0
        skipped_count = 0
        
        for (parent_asin,) in parent_asins:
            # 从meta表获取信息
            meta_query = """
                SELECT title, main_category, features, description, details
                FROM meta
                WHERE parent_asin = ?
                LIMIT 1
            """
            meta_results = self.db.execute_read(meta_query, {"parent_asin": parent_asin})
            
            if not meta_results:
                logger.warning(f"未找到 parent_asin={parent_asin} 的元数据，跳过")
                skipped_count += 1
                continue
            
            meta_row = meta_results[0]
            product_title = meta_row[0] or ""
            main_category = meta_row[1] or ""
            features = meta_row[2] or ""
            description = meta_row[3] or ""
            details = meta_row[4] or ""
            
            # 截断处理
            features_short = self._truncate_text(features, self.max_length)
            description_short = self._truncate_text(description, self.max_length)
            details_short = self._truncate_text(details, self.max_length)
            
            # 插入meta_context
            insert_query = f"""
                INSERT INTO {table_manager.META_CONTEXT}
                (run_id, pipeline_version, data_slice_id, created_at,
                 parent_asin, product_title, main_category,
                 features_short, description_short, details_short, context_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # 使用位置参数列表
            self.db.execute_write(insert_query, [
                self.run_id,
                self.pipeline_version,
                self.data_slice_id,
                self.version_fields["created_at"],
                parent_asin,
                product_title,
                main_category,
                features_short,
                description_short,
                details_short,
                self.context_version
            ])
            inserted_count += 1
            
            # 每10条记录输出一次进度
            if inserted_count % 10 == 0:
                logger.info(f"已处理 {inserted_count} 个商品的元数据上下文...")
        
        logger.info(
            f"MetaContextAgent完成: 处理={inserted_count}, 跳过={skipped_count}"
        )
        
        return {
            "status": "success",
            "processed_count": inserted_count,
            "table": table_manager.META_CONTEXT
        }
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断文本"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

