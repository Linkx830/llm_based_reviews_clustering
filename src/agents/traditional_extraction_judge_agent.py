# 新增或修改我时需要修改这个文件夹中的README.md文件
"""TraditionalExtractionJudgeAgent - 传统抽取校验、归一、噪声处理"""
import json
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.normalizer import TextNormalizer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TraditionalExtractionJudgeAgent(BaseAgent):
    """
    TraditionalExtractionJudgeAgent
    
    职责：
    - 对传统抽取结果做统一质量门禁
    - aspect同义归一（taxonomy）
    - issue归一（轻量同义/词形）
    - 噪声/无效记录过滤（NOISE/INVALID）
    - rating弱一致性冲突标记（quality_flags）
    
    输入表：aspect_sentiment_raw_traditional, review_sentences
    输出表：aspect_sentiment_valid_traditional, extraction_issues_traditional
    """
    
    def __init__(
        self, 
        *args, 
        embedding_tool=None,
        use_semantic_matching: bool = False,
        semantic_threshold: float = 0.7,
        **kwargs
    ):
        """
        Args:
            embedding_tool: Embedding工具（可选，用于语义相似度匹配）
            use_semantic_matching: 是否启用语义相似度匹配（需要embedding_tool）
            semantic_threshold: 语义相似度阈值（默认0.7）
        """
        super().__init__(*args, **kwargs)
        self.normalizer = TextNormalizer(
            embedding_tool=embedding_tool,
            use_semantic_matching=use_semantic_matching,
            semantic_threshold=semantic_threshold,
            llm_wrapper=None,  # 传统方法不使用LLM
            use_llm_matching=False
        )
    
    def process(self) -> Dict[str, Any]:
        """执行校验与归一"""
        logger.info(f"开始执行TraditionalExtractionJudgeAgent，run_id={self.run_id}")
        
        # 读取传统抽取结果
        query = f"""
            SELECT asr.sentence_id, asr.aspect_raw, asr.issue_raw,
                   asr.sentiment, asr.sentiment_score, asr.evidence_text,
                   rs.review_pk, rs.parent_asin, rs.timestamp, rs.rating,
                   rs.target_sentence
            FROM {TableManager.ASPECT_SENTIMENT_RAW_TRADITIONAL} asr
            JOIN {TableManager.REVIEW_SENTENCES} rs 
                ON asr.sentence_id = rs.sentence_id
            WHERE asr.run_id = ?
            QUALIFY ROW_NUMBER() OVER (PARTITION BY asr.sentence_id, asr.aspect_raw ORDER BY asr.created_at DESC) = 1
        """
        raw_results = self.db.execute_read(query, {"run_id": self.run_id})
        logger.info(f"读取到 {len(raw_results)} 条传统抽取记录（已去重）")
        
        # 额外检查：使用Python去重作为备选
        seen_keys = set()
        unique_results = []
        duplicate_count = 0
        for row in raw_results:
            sentence_id = row[0]
            aspect_raw = row[1]
            key = (sentence_id, aspect_raw)
            if key in seen_keys:
                duplicate_count += 1
                logger.warning(f"发现重复记录: sentence_id={sentence_id}, aspect_raw={aspect_raw}，跳过")
                continue
            seen_keys.add(key)
            unique_results.append(row)
        
        if duplicate_count > 0:
            logger.warning(f"⚠️ 发现 {duplicate_count} 个重复记录，已跳过")
            raw_results = unique_results
            logger.info(f"Python去重后，实际处理 {len(raw_results)} 条记录")
        
        table_manager = TableManager(self.db)
        valid_count = 0
        noise_count = 0
        invalid_count = 0
        total_raw_aspect_count = len(raw_results)
        
        for row in raw_results:
            sentence_id, aspect_raw, issue_raw, sentiment, sentiment_score, evidence_text, \
                review_pk, parent_asin, timestamp, rating, target_sentence = row
            
            try:
                # 校验证据可定位性
                if not self._validate_evidence(evidence_text, target_sentence):
                    self._record_issue(sentence_id, "EVIDENCE_NOT_FOUND", "证据无法定位")
                    invalid_count += 1
                    continue
                
                # 归一化aspect和issue
                aspect_norm = self.normalizer.normalize_aspect(aspect_raw)
                issue_norm = self.normalizer.normalize_issue(issue_raw)
                
                # 判断是否为噪声
                validity_label = self.normalizer.judge_validity(aspect_norm, issue_norm)
                
                if validity_label == "INVALID":
                    invalid_count += 1
                    continue
                elif validity_label == "NOISE":
                    noise_count += 1
                    continue
                
                # 检查rating与sentiment的一致性
                quality_flags = []
                if self._check_rating_sentiment_conflict(rating, sentiment):
                    quality_flags.append("RATING_CONFLICT")
                
                # 检查是否已存在相同的记录
                check_query = f"""
                    SELECT COUNT(*) FROM {table_manager.ASPECT_SENTIMENT_VALID_TRADITIONAL}
                    WHERE run_id = ? AND sentence_id = ? 
                      AND aspect_norm = ? AND issue_norm = ?
                      AND validity_label = 'VALID'
                """
                existing = self.db.execute_read(check_query, {
                    "run_id": self.run_id,
                    "sentence_id": sentence_id,
                    "aspect_norm": aspect_norm,
                    "issue_norm": issue_norm
                })
                
                if existing[0][0] > 0:
                    logger.debug(
                        f"跳过重复记录: sentence_id={sentence_id}, "
                        f"aspect_norm={aspect_norm}, issue_norm={issue_norm}"
                    )
                    continue
                
                # 插入valid表
                insert_query = f"""
                    INSERT INTO {table_manager.ASPECT_SENTIMENT_VALID_TRADITIONAL}
                    (run_id, pipeline_version, data_slice_id, created_at,
                     sentence_id, review_pk, parent_asin, timestamp, rating,
                     aspect_raw, aspect_norm, sentiment, sentiment_score,
                     issue_raw, issue_norm, evidence_text, validity_label, quality_flags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                quality_flags_str = ",".join(quality_flags) if quality_flags else None
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
                    "aspect_raw": aspect_raw,
                    "aspect_norm": aspect_norm,
                    "sentiment": sentiment or "neutral",
                    "sentiment_score": sentiment_score or 0.0,
                    "issue_raw": issue_raw,
                    "issue_norm": issue_norm,
                    "evidence_text": evidence_text,
                    "validity_label": validity_label,
                    "quality_flags": quality_flags_str
                })
                valid_count += 1
            except Exception as e:
                logger.error(f"处理sentence_id={sentence_id}时出错: {str(e)}")
                self._record_issue(sentence_id, "PARSE_FAIL", str(e))
                invalid_count += 1
        
        logger.info(
            f"TraditionalExtractionJudgeAgent完成: "
            f"原始aspect数量={total_raw_aspect_count}, "
            f"valid={valid_count}, noise={noise_count}, invalid={invalid_count}, "
            f"有效率={valid_count/total_raw_aspect_count*100:.1f}%" if total_raw_aspect_count > 0 else "有效率=N/A"
        )
        
        return {
            "status": "success",
            "total_raw_aspect_count": total_raw_aspect_count,
            "valid_count": valid_count,
            "noise_count": noise_count,
            "invalid_count": invalid_count,
            "table": table_manager.ASPECT_SENTIMENT_VALID_TRADITIONAL
        }
    
    def _validate_evidence(self, evidence: str, target_sentence: str) -> bool:
        """验证证据可定位性"""
        return evidence.lower() in target_sentence.lower()
    
    def _check_rating_sentiment_conflict(self, rating: float, sentiment: str) -> bool:
        """检查rating与sentiment的一致性冲突"""
        if rating is None or sentiment is None:
            return False
        
        # rating >= 4 但sentiment为negative，或 rating <= 2 但sentiment为positive
        if rating >= 4.0 and sentiment == "negative":
            return True
        if rating <= 2.0 and sentiment == "positive":
            return True
        
        return False
    
    def _record_issue(self, sentence_id: str, issue_type: str, details: str):
        """记录问题"""
        insert_query = f"""
            INSERT INTO {TableManager.EXTRACTION_ISSUES_TRADITIONAL}
            (run_id, pipeline_version, data_slice_id, created_at,
             sentence_id, issue_type, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute_write(insert_query, {
            "run_id": self.run_id,
            "pipeline_version": self.pipeline_version,
            "data_slice_id": self.data_slice_id,
            "created_at": self.version_fields["created_at"],
            "sentence_id": sentence_id,
            "issue_type": issue_type,
            "details": details
        })



