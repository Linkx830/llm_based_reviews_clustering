# 新增或修改我时需要修改这个文件夹中的README.md文件
"""PreprocessAgent - 评论级清洗与文本规范化"""
import re
from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessAgent(BaseAgent):
    """
    PreprocessAgent
    
    职责：
    - 合并文本：review_title + review_text
    - 去空/去短/去重复
    - 规范化字符与空白
    
    输入表：selected_reviews
    输出表：normalized_reviews
    """
    
    def process(self) -> Dict[str, Any]:
        """
        预处理评论
        
        Returns:
            处理结果统计
        """
        logger.info(f"开始执行PreprocessAgent，run_id={self.run_id}")
        
        # 读取selected_reviews
        query = f"""
            SELECT review_pk, parent_asin, timestamp, rating,
                   review_title, review_text
            FROM {TableManager.SELECTED_REVIEWS}
            WHERE run_id = ?
        """
        reviews = self.db.execute_read(query, {"run_id": self.run_id})
        logger.info(f"读取到 {len(reviews)} 条评论")
        
        table_manager = TableManager(self.db)
        processed_count = 0
        skipped_count = 0
        
        for row in reviews:
            review_pk, parent_asin, timestamp, rating, review_title, review_text = row
            
            # 合并文本
            clean_text = self._merge_and_clean(review_title, review_text)
            
            # 跳过过短的文本
            if len(clean_text.strip()) < 10:
                skipped_count += 1
                continue
            
            # 记录清洗标志
            cleaning_flags = []
            if not review_title:
                cleaning_flags.append("no_title")
            
            # 插入normalized_reviews
            insert_query = f"""
                INSERT INTO {table_manager.NORMALIZED_REVIEWS}
                (run_id, pipeline_version, data_slice_id, created_at,
                 review_pk, parent_asin, timestamp, rating,
                 clean_text, cleaning_flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.db.execute_write(insert_query, {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id,
                "created_at": self.version_fields["created_at"],
                "review_pk": review_pk,
                "parent_asin": parent_asin,
                "timestamp": timestamp,
                "rating": rating,
                "clean_text": clean_text,
                "cleaning_flags": ",".join(cleaning_flags) if cleaning_flags else None
            })
            processed_count += 1
            
            # 每100条记录输出一次进度
            if processed_count % 100 == 0:
                logger.info(f"已处理 {processed_count} 条评论...")
        
        logger.info(
            f"PreprocessAgent完成: 处理={processed_count}, 跳过={skipped_count}"
        )
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "table": table_manager.NORMALIZED_REVIEWS
        }
    
    def _merge_and_clean(self, title: str, text: str) -> str:
        """合并并清洗文本"""
        # 合并
        parts = []
        if title:
            parts.append(title.strip())
        if text:
            parts.append(text.strip())
        
        merged = " ".join(parts)
        
        # 规范化空白
        merged = re.sub(r'\s+', ' ', merged)
        
        # 去除HTML标签（简单处理）
        merged = re.sub(r'<[^>]+>', '', merged)
        
        return merged.strip()

