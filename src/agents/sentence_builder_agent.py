# 新增或修改我时需要修改这个文件夹中的README.md文件
"""SentenceBuilderAgent - 句子构建 + 上下文窗口"""
import re
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SentenceBuilderAgent(BaseAgent):
    """
    SentenceBuilderAgent
    
    职责：
    - 将评论拆成句子，并为每个句子构建上下文窗口
    - 输出目标句与上下文同时存在的结构
    
    输入表：normalized_reviews
    输出表：review_sentences
    """
    
    def process(self, context_window: int = 1) -> Dict[str, Any]:
        """
        构建句子表
        
        Args:
            context_window: 上下文窗口大小（前后各取几句）
        
        Returns:
            处理结果统计
        """
        logger.info(
            f"开始执行SentenceBuilderAgent，run_id={self.run_id}, "
            f"context_window={context_window}"
        )
        
        # 读取normalized_reviews
        query = f"""
            SELECT review_pk, parent_asin, timestamp, rating,
                   clean_text
            FROM {TableManager.NORMALIZED_REVIEWS}
            WHERE run_id = ?
        """
        reviews = self.db.execute_read(query, {"run_id": self.run_id})
        logger.info(f"读取到 {len(reviews)} 条规范化评论")
        
        table_manager = TableManager(self.db)
        sentence_count = 0
        review_count = 0
        
        for row in reviews:
            review_pk, parent_asin, timestamp, rating, clean_text = row
            
            # 分句
            sentences = self._split_sentences(clean_text)
            
            # 为每个句子构建上下文
            for idx, target_sentence in enumerate(sentences):
                # 获取上下文
                prev_sentences = sentences[max(0, idx - context_window):idx]
                next_sentences = sentences[idx + 1:idx + 1 + context_window]
                
                prev_sentence = prev_sentences[-1] if prev_sentences else None
                next_sentence = next_sentences[0] if next_sentences else None
                
                # 构建context_text
                context_parts = []
                if prev_sentence:
                    context_parts.append(prev_sentence)
                context_parts.append(target_sentence)
                if next_sentence:
                    context_parts.append(next_sentence)
                context_text = " ".join(context_parts)
                
                # 生成sentence_id
                sentence_id = f"{review_pk}_{idx}"
                
                # 插入review_sentences
                insert_query = f"""
                    INSERT INTO {table_manager.REVIEW_SENTENCES}
                    (run_id, pipeline_version, data_slice_id, created_at,
                     sentence_id, review_pk, parent_asin, timestamp, rating,
                     sentence_index, target_sentence, prev_sentence,
                     next_sentence, context_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                self.db.execute_write(insert_query, {
                    "run_id": self.run_id,
                    "pipeline_version": self.pipeline_version,
                    "data_slice_id": self.data_slice_id,
                    "created_at": self.version_fields["created_at"],
                    "sentence_id": sentence_id,
                    "review_pk": review_pk,
                    "parent_asin": parent_asin,
                    "timestamp": timestamp,
                    "rating": rating,
                    "sentence_index": idx,
                    "target_sentence": target_sentence,
                    "prev_sentence": prev_sentence,
                    "next_sentence": next_sentence,
                    "context_text": context_text
                })
                sentence_count += 1
            
            review_count += 1
            # 每50条评论输出一次进度
            if review_count % 50 == 0:
                logger.info(
                    f"已处理 {review_count}/{len(reviews)} 条评论，"
                    f"生成 {sentence_count} 个句子..."
                )
        
        logger.info(
            f"SentenceBuilderAgent完成: 处理了 {review_count} 条评论，"
            f"生成 {sentence_count} 个句子"
        )
        
        return {
            "status": "success",
            "sentence_count": sentence_count,
            "table": table_manager.REVIEW_SENTENCES
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句（简单实现）"""
        # 使用句号、问号、感叹号分句
        sentences = re.split(r'[.!?]+', text)
        # 过滤空句和过短句
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        return sentences

