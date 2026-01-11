# 新增或修改我时需要修改这个文件夹中的README.md文件
"""OpinionCandidateFilterAgent - 观点候选句过滤（成本控制）"""
import re
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OpinionCandidateFilterAgent(BaseAgent):
    """
    OpinionCandidateFilterAgent
    
    职责：
    - 过滤掉信息密度低/不含观点/纯客观的句子
    - 仅保留"疑似含观点"的候选句进入LLM抽取
    - 使用评分系统，更全面地识别鼓励、批评、建议等观点表达
    
    输入表：review_sentences
    输出表：opinion_candidates
    """
    
    # 评价形容词/副词关键词（强信号，权重高）
    OPINION_KEYWORDS_STRONG = [
        "good", "bad", "great", "terrible", "excellent", "awful",
        "nice", "horrible", "wonderful", "poor", "amazing", "disappointing",
        "love", "hate", "like", "dislike", "perfect", "flawed",
        "fantastic", "awful", "brilliant", "mediocre", "outstanding", "pathetic",
        "satisfied", "disappointed", "pleased", "frustrated", "impressed", "unimpressed"
    ]
    
    # 建议类关键词（鼓励、建议、推荐）
    SUGGESTION_KEYWORDS = [
        "should", "recommend", "suggest", "advice", "tip", "better",
        "prefer", "wish", "hope", "expect", "would", "could",
        "consider", "try", "avoid", "worth", "worthwhile"
    ]
    
    # 比较类关键词
    COMPARISON_KEYWORDS = [
        "better", "worse", "compared", "than", "versus", "vs",
        "similar", "different", "same", "instead", "rather",
        "prefer", "preferable", "superior", "inferior"
    ]
    
    # 问题描述类关键词
    PROBLEM_KEYWORDS = [
        "problem", "issue", "defect", "flaw", "broken", "faulty",
        "malfunction", "error", "bug", "glitch", "failure", "failed",
        "doesn't work", "won't work", "stopped working", "not working"
    ]
    
    # 功能评价类关键词
    FUNCTION_KEYWORDS = [
        "works", "functions", "performs", "operates", "runs",
        "handles", "manages", "supports", "provides", "delivers",
        "meets", "exceeds", "fails", "lacks", "missing"
    ]
    
    # 经验分享类关键词
    EXPERIENCE_KEYWORDS = [
        "after", "since", "when", "used", "tried", "tested",
        "bought", "purchased", "owned", "have had", "been using",
        "for months", "for years", "so far", "up to now"
    ]
    
    # 否定词
    NEGATION_WORDS = ["not", "never", "no", "none", "nothing", "nobody", "neither", "nor"]
    
    # 程度词
    INTENSITY_WORDS = ["very", "extremely", "quite", "really", "too", "so", "pretty", "rather"]
    
    # 观点表达模式（正则表达式）
    OPINION_PATTERNS = [
        # 建议模式
        (r'\b(should|would|could|might)\s+\w+', "suggestion_pattern"),
        (r'\b(recommend|suggest|advise)\s+\w+', "recommendation_pattern"),
        (r'\b(wish|hope|expect)\s+\w+', "expectation_pattern"),
        # 比较模式
        (r'\b(better|worse|best|worst)\s+than', "comparison_pattern"),
        (r'\b(more|less)\s+\w+\s+than', "comparative_pattern"),
        # 问题模式
        (r'\b(problem|issue|defect|flaw)\s+with', "problem_pattern"),
        (r'\b(doesn\'t|won\'t|can\'t)\s+\w+', "negation_verb_pattern"),
        # 评价模式
        (r'\b(too|so|very|extremely)\s+\w+', "intensity_pattern"),
        (r'\b(not|never)\s+\w+\s+(enough|good|bad)', "negation_evaluation_pattern"),
        # 经验模式
        (r'\b(after|since)\s+\w+\s+(months?|years?|days?|weeks?)', "time_experience_pattern"),
        (r'\b(been|have been|has been)\s+\w+ing', "continuous_experience_pattern"),
    ]
    
    def process(
        self,
        min_length: int = 10,
        max_length: int = 500,
        score_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        过滤观点候选句
        
        Args:
            min_length: 最小句长
            max_length: 最大句长
            score_threshold: 观点分数阈值，达到此分数才被认为是候选句
        
        Returns:
            处理结果统计
        """
        logger.info(
            f"开始执行OpinionCandidateFilterAgent，run_id={self.run_id}, "
            f"min_length={min_length}, max_length={max_length}, "
            f"score_threshold={score_threshold}"
        )
        
        # 读取review_sentences
        query = f"""
            SELECT sentence_id, target_sentence
            FROM {TableManager.REVIEW_SENTENCES}
            WHERE run_id = ?
        """
        sentences = self.db.execute_read(query, {"run_id": self.run_id})
        logger.info(f"读取到 {len(sentences)} 个句子")
        
        table_manager = TableManager(self.db)
        candidate_count = 0
        filtered_count = 0
        score_distribution = {}  # 用于统计分数分布
        
        for row in sentences:
            sentence_id, target_sentence = row
            
            # 应用过滤规则（使用评分系统）
            is_candidate, reason, score = self._filter_sentence(
                target_sentence, min_length, max_length, score_threshold
            )
            
            # 统计分数分布
            score_bucket = int(score)
            score_distribution[score_bucket] = score_distribution.get(score_bucket, 0) + 1
            
            if is_candidate:
                candidate_count += 1
            else:
                filtered_count += 1
            
            # 插入opinion_candidates（添加priority_weight字段）
            insert_query = f"""
                INSERT INTO {table_manager.OPINION_CANDIDATES}
                (run_id, pipeline_version, data_slice_id, created_at,
                 sentence_id, is_candidate, filter_reason, priority_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.db.execute_write(insert_query, {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id,
                "created_at": self.version_fields["created_at"],
                "sentence_id": sentence_id,
                "is_candidate": is_candidate,
                "filter_reason": reason,
                "priority_weight": score
            })
        
        logger.info(
            f"OpinionCandidateFilterAgent完成: 候选={candidate_count}, "
            f"过滤={filtered_count}, 候选率={candidate_count/len(sentences)*100:.2f}%"
        )
        logger.debug(f"分数分布: {score_distribution}")
        
        return {
            "status": "success",
            "candidate_count": candidate_count,
            "filtered_count": filtered_count,
            "score_threshold": score_threshold,
            "table": table_manager.OPINION_CANDIDATES
        }
    
    def _filter_sentence(
        self, 
        sentence: str, 
        min_length: int, 
        max_length: int,
        score_threshold: float
    ) -> Tuple[bool, str, float]:
        """
        判断句子是否为观点候选（使用评分系统）
        
        Returns:
            (is_candidate, reason, score)
        """
        sentence_lower = sentence.lower()
        score = 0.0
        reasons = []
        
        # 长度检查（硬约束）
        if len(sentence) < min_length:
            return False, "too_short", 0.0
        if len(sentence) > max_length:
            return False, "too_long", 0.0
        
        # 1. 强评价关键词（权重：2.0）
        strong_opinion_count = sum(
            1 for keyword in self.OPINION_KEYWORDS_STRONG 
            if keyword in sentence_lower
        )
        if strong_opinion_count > 0:
            score += min(strong_opinion_count * 2.0, 4.0)  # 最多4分
            reasons.append(f"strong_opinion({strong_opinion_count})")
        
        # 2. 建议类关键词（权重：1.5）
        suggestion_count = sum(
            1 for keyword in self.SUGGESTION_KEYWORDS 
            if keyword in sentence_lower
        )
        if suggestion_count > 0:
            score += min(suggestion_count * 1.5, 3.0)  # 最多3分
            reasons.append(f"suggestion({suggestion_count})")
        
        # 3. 比较类关键词（权重：1.5）
        comparison_count = sum(
            1 for keyword in self.COMPARISON_KEYWORDS 
            if keyword in sentence_lower
        )
        if comparison_count > 0:
            score += min(comparison_count * 1.5, 3.0)  # 最多3分
            reasons.append(f"comparison({comparison_count})")
        
        # 4. 问题描述类关键词（权重：1.5）
        problem_count = sum(
            1 for keyword in self.PROBLEM_KEYWORDS 
            if keyword in sentence_lower
        )
        if problem_count > 0:
            score += min(problem_count * 1.5, 3.0)  # 最多3分
            reasons.append(f"problem({problem_count})")
        
        # 5. 功能评价类关键词（权重：1.0）
        function_count = sum(
            1 for keyword in self.FUNCTION_KEYWORDS 
            if keyword in sentence_lower
        )
        if function_count > 0:
            score += min(function_count * 1.0, 2.0)  # 最多2分
            reasons.append(f"function({function_count})")
        
        # 6. 经验分享类关键词（权重：1.0）
        experience_count = sum(
            1 for keyword in self.EXPERIENCE_KEYWORDS 
            if keyword in sentence_lower
        )
        if experience_count > 0:
            score += min(experience_count * 1.0, 2.0)  # 最多2分
            reasons.append(f"experience({experience_count})")
        
        # 7. 否定词 + 程度词组合（权重：1.5）
        has_negation = any(word in sentence_lower for word in self.NEGATION_WORDS)
        has_intensity = any(word in sentence_lower for word in self.INTENSITY_WORDS)
        if has_negation and has_intensity:
            score += 1.5
            reasons.append("negation_intensity")
        elif has_negation:
            score += 0.5
            reasons.append("negation")
        
        # 8. 感叹号（权重：0.5）
        if "!" in sentence:
            score += 0.5
            reasons.append("exclamation")
        
        # 9. 模式匹配（权重：1.0-2.0）
        pattern_matches = []
        for pattern, pattern_name in self.OPINION_PATTERNS:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                pattern_matches.append(pattern_name)
                # 不同模式权重不同
                if "suggestion" in pattern_name or "recommendation" in pattern_name:
                    score += 1.5
                elif "problem" in pattern_name or "negation" in pattern_name:
                    score += 1.5
                elif "comparison" in pattern_name:
                    score += 1.0
                else:
                    score += 1.0
        
        if pattern_matches:
            reasons.append(f"patterns({','.join(pattern_matches[:3])})")  # 最多显示3个
        
        # 10. 句子结构特征
        # 包含第一人称（更可能是个人观点）
        if re.search(r'\b(i|my|me|we|our)\b', sentence_lower):
            score += 0.5
            reasons.append("first_person")
        
        # 包含疑问句（可能是建议或问题）
        if "?" in sentence:
            score += 0.3
            reasons.append("question")
        
        # 综合判断
        reason_str = "+".join(reasons) if reasons else "no_opinion_signal"
        
        if score >= score_threshold:
            return True, reason_str, score
        else:
            return False, reason_str, score

