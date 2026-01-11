# 新增或修改我时需要修改这个文件夹中的README.md文件
"""ExtractionJudgeAgent - 抽取校验、归一、噪声处理"""
import json
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..tools.structured_output_tool import StructuredOutputTool
from ..utils.normalizer import TextNormalizer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExtractionJudgeAgent(BaseAgent):
    """
    ExtractionJudgeAgent
    
    职责：
    - 校验抽取结果结构与证据可定位性
    - aspect同义归一（支持多种匹配策略：精确、部分、模糊、关键词、语义相似度、LLM辅助匹配）
    - sentiment与rating弱一致性冲突标记
    - 噪声/无效类识别与剔除
    
    输入表：aspect_sentiment_raw, review_sentences
    输出表：aspect_sentiment_valid, extraction_issues
    """
    
    def __init__(
        self, 
        *args, 
        embedding_tool=None,
        use_semantic_matching: bool = False,
        semantic_threshold: float = 0.7,
        llm_wrapper=None,
        use_llm_matching: bool = False,
        llm_confidence_threshold: float = 0.5,
        **kwargs
    ):
        """
        Args:
            embedding_tool: Embedding工具（可选，用于语义相似度匹配）
            use_semantic_matching: 是否启用语义相似度匹配（需要embedding_tool）
            semantic_threshold: 语义相似度阈值（默认0.7）
            llm_wrapper: LLM包装器（可选，用于LLM辅助匹配）
            use_llm_matching: 是否启用LLM辅助匹配（需要llm_wrapper）
            llm_confidence_threshold: LLM匹配的置信度阈值（默认0.5）
        """
        super().__init__(*args, **kwargs)
        self.normalizer = TextNormalizer(
            embedding_tool=embedding_tool,
            use_semantic_matching=use_semantic_matching,
            semantic_threshold=semantic_threshold,
            llm_wrapper=llm_wrapper,
            use_llm_matching=use_llm_matching,
            llm_confidence_threshold=llm_confidence_threshold
        )
    
    def process(self) -> Dict[str, Any]:
        """执行校验与归一"""
        logger.info(f"开始执行ExtractionJudgeAgent，run_id={self.run_id}")
        
        # 读取成功抽取的结果
        # 注意：使用ROW_NUMBER()去重，避免重复处理同一个sentence_id（断点续跑可能导致重复）
        query = f"""
            SELECT asr.sentence_id, asr.llm_output,
                   rs.review_pk, rs.parent_asin, rs.timestamp, rs.rating,
                   rs.target_sentence
            FROM {TableManager.ASPECT_SENTIMENT_RAW} asr
            JOIN {TableManager.REVIEW_SENTENCES} rs 
                ON asr.sentence_id = rs.sentence_id
            WHERE asr.run_id = ? AND asr.parse_status = 'SUCCESS'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY asr.sentence_id ORDER BY asr.created_at DESC) = 1
        """
        raw_results = self.db.execute_read(query, {"run_id": self.run_id})
        logger.info(f"读取到 {len(raw_results)} 条成功抽取的记录（已去重）")
        
        # 额外检查：使用Python去重作为备选（如果QUALIFY不支持）
        seen_sentence_ids = set()
        unique_results = []
        duplicate_count = 0
        for row in raw_results:
            sentence_id = row[0]
            if sentence_id in seen_sentence_ids:
                duplicate_count += 1
                logger.warning(f"发现重复的sentence_id: {sentence_id}，跳过重复处理")
                continue
            seen_sentence_ids.add(sentence_id)
            unique_results.append(row)
        
        if duplicate_count > 0:
            logger.warning(
                f"⚠️ 发现 {duplicate_count} 个重复的sentence_id，已跳过。"
                f"这可能是因为aspect_sentiment_raw表中有重复记录（断点续跑导致）。"
            )
            raw_results = unique_results
            logger.info(f"Python去重后，实际处理 {len(raw_results)} 条记录")
        
        table_manager = TableManager(self.db)
        valid_count = 0
        noise_count = 0
        invalid_count = 0
        total_raw_aspect_count = 0  # 统计原始aspect数量（处理前）
        
        for row in raw_results:
            sentence_id, llm_output_json, review_pk, parent_asin, timestamp, rating, target_sentence = row
            
            try:
                output_data = json.loads(llm_output_json)
                aspects = output_data.get("aspects", [])
                total_raw_aspect_count += len(aspects)
                
                for aspect_data in aspects:
                    # 校验证据可定位性
                    evidence_text = aspect_data.get("evidence_text", "")
                    if not self._validate_evidence(evidence_text, target_sentence):
                        self._record_issue(sentence_id, "EVIDENCE_NOT_FOUND", "证据无法定位")
                        invalid_count += 1
                        continue
                    
                    # 归一化aspect和issue
                    aspect_raw = aspect_data.get("aspect", "")
                    issue_raw = aspect_data.get("issue", "")
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
                    
                    # 检查是否已存在相同的记录（避免重复插入）
                    # 注意：这里只检查sentence_id+aspect_norm+issue_norm的组合，因为同一个句子可能有多个不同的aspect
                    check_query = f"""
                        SELECT COUNT(*) FROM {table_manager.ASPECT_SENTIMENT_VALID}
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
                        INSERT INTO {table_manager.ASPECT_SENTIMENT_VALID}
                        (run_id, pipeline_version, data_slice_id, created_at,
                         sentence_id, review_pk, parent_asin, timestamp, rating,
                         aspect_raw, aspect_norm, sentiment, sentiment_score,
                         issue_raw, issue_norm, evidence_text, validity_label)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        "aspect_raw": aspect_raw,
                        "aspect_norm": aspect_norm,
                        "sentiment": aspect_data.get("sentiment", "neutral"),
                        "sentiment_score": aspect_data.get("sentiment_score", 0.0),
                        "issue_raw": issue_raw,
                        "issue_norm": issue_norm,
                        "evidence_text": evidence_text,
                        "validity_label": validity_label
                    })
                    valid_count += 1
            except Exception as e:
                logger.error(f"处理sentence_id={sentence_id}时出错: {str(e)}")
                self._record_issue(sentence_id, "PARSE_FAIL", str(e))
                invalid_count += 1
        
        # 获取LLM新增的aspect（如果启用了LLM匹配）
        new_aspects = self.normalizer.get_new_aspects()
        dynamic_synonyms = self.normalizer.get_dynamic_synonyms()
        
        if new_aspects:
            logger.info(
                f"LLM新增了 {len(new_aspects)} 个aspect到词表: {sorted(new_aspects)}"
            )
            logger.info(
                f"动态同义词映射数量: {len(dynamic_synonyms)}"
            )
        
        logger.info(
            f"ExtractionJudgeAgent完成: "
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
            "table": table_manager.ASPECT_SENTIMENT_VALID,
            "new_aspects": list(new_aspects),
            "dynamic_synonyms": dynamic_synonyms
        }
    
    def _validate_evidence(self, evidence: str, target_sentence: str) -> bool:
        """验证证据可定位性"""
        return evidence.lower() in target_sentence.lower()
    
    
    def _record_issue(self, sentence_id: str, issue_type: str, details: str):
        """记录问题"""
        insert_query = f"""
            INSERT INTO {TableManager.EXTRACTION_ISSUES}
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

