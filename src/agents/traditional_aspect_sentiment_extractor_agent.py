# 新增或修改我时需要修改这个文件夹中的README.md文件
"""TraditionalAspectSentimentExtractorAgent - 传统NLP抽取（无LLM）"""
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TraditionalAspectSentimentExtractorAgent(BaseAgent):
    """
    TraditionalAspectSentimentExtractorAgent - 传统NLP抽取
    
    职责：
    - 使用规则/词典/统计特征从句子中抽取aspect和sentiment
    - 不使用LLM，完全基于传统NLP方法
    
    输入表：review_sentences, opinion_candidates, meta_context
    输出表：aspect_sentiment_raw_traditional
    """
    
    def __init__(
        self,
        *args,
        extract_method: str = "LEXICON_RULE",
        aspect_seed_lexicon: Optional[List[str]] = None,
        use_meta_context: bool = True,
        max_candidates_per_sentence: int = 5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.extract_method = extract_method
        self.use_meta_context = use_meta_context
        self.max_candidates_per_sentence = max_candidates_per_sentence
        
        # 初始化情感词典
        self._init_sentiment_lexicons()
        
        # 初始化aspect种子词表
        if aspect_seed_lexicon is None:
            self.aspect_seed_lexicon = {
                "battery", "screen", "size", "quality", "price", "design",
                "performance", "speed", "weight", "durability", "sound",
                "camera", "display", "storage", "memory", "processor",
                "build", "material", "comfort", "usability", "interface"
            }
        else:
            self.aspect_seed_lexicon = set(aspect_seed_lexicon)
        
        # 停用名词（泛化词）
        self.stop_nouns = {
            "product", "item", "thing", "it", "this", "that", "one",
            "review", "amazon", "seller", "company", "brand"
        }
    
    def _init_sentiment_lexicons(self):
        """初始化情感词典"""
        # 正面情感词
        self.positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "perfect", "love", "like", "enjoy", "satisfied", "happy",
            "pleased", "impressed", "outstanding", "superb", "brilliant",
            "awesome", "best", "better", "nice", "fine", "decent"
        }
        
        # 负面情感词
        self.negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "hate",
            "disappointed", "disappointing", "poor", "cheap", "broken",
            "defective", "faulty", "useless", "waste", "regret", "return",
            "refund", "problem", "issue", "flaw", "defect", "damaged"
        }
        
        # 否定词
        self.negation_words = {
            "not", "no", "never", "none", "neither", "nobody", "nothing",
            "nowhere", "without", "lack", "missing", "fail", "cannot"
        }
        
        # 程度词
        self.intensity_words = {
            "very", "extremely", "highly", "quite", "really", "truly",
            "absolutely", "completely", "totally", "utterly", "incredibly",
            "somewhat", "rather", "pretty", "fairly", "slightly", "too"
        }
    
    def process(self) -> Dict[str, Any]:
        """执行传统抽取"""
        logger.info(f"开始执行TraditionalAspectSentimentExtractorAgent，run_id={self.run_id}")
        
        # 获取候选句子
        query = f"""
            SELECT rs.sentence_id, rs.target_sentence, rs.context_text,
                   rs.parent_asin, mc.product_title, mc.features_short
            FROM {TableManager.REVIEW_SENTENCES} rs
            JOIN {TableManager.OPINION_CANDIDATES} oc 
                ON rs.sentence_id = oc.sentence_id 
                AND rs.run_id = oc.run_id
            LEFT JOIN {TableManager.META_CONTEXT} mc 
                ON rs.parent_asin = mc.parent_asin AND mc.run_id = ?
            WHERE rs.run_id = ? AND oc.is_candidate = true
        """
        candidates = self.db.execute_read(query, {
            "run_id": self.run_id,
            "run_id2": self.run_id
        })
        
        logger.info(f"读取到 {len(candidates)} 条候选句子")
        
        table_manager = TableManager(self.db)
        success_count = 0
        fail_count = 0
        total_aspect_count = 0
        
        for row in candidates:
            sentence_id, target_sentence, context_text, parent_asin, product_title, features_short = row
            
            try:
                # 阶段A-C：抽取aspect和sentiment
                aspects = self._extract_aspects_and_sentiment(
                    target_sentence, context_text, product_title, features_short
                )
                
                if not aspects:
                    fail_count += 1
                    logger.debug(f"sentence_id={sentence_id} 未抽取到aspect")
                    continue
                
                total_aspect_count += len(aspects)
                
                # 为每个aspect生成记录
                for aspect_data in aspects:
                    # 阶段D：抽取issue短语
                    issue_raw = self._extract_issue_phrase(
                        target_sentence, aspect_data["aspect_raw"], aspect_data["evidence_text"]
                    )
                    aspect_data["issue_raw"] = issue_raw
                    
                    # 阶段E：验证evidence可定位性
                    if not self._validate_evidence(aspect_data["evidence_text"], target_sentence):
                        logger.warning(
                            f"sentence_id={sentence_id} evidence无法定位: {aspect_data['evidence_text']}"
                        )
                        # 如果evidence无法定位，尝试从target_sentence中重新提取
                        aspect_data["evidence_text"] = self._extract_evidence_from_sentence(
                            target_sentence, aspect_data["aspect_raw"]
                        )
                    
                    # 插入aspect_sentiment_raw_traditional
                    insert_query = f"""
                        INSERT INTO {table_manager.ASPECT_SENTIMENT_RAW_TRADITIONAL}
                        (run_id, pipeline_version, data_slice_id, created_at,
                         sentence_id, extract_method, aspect_raw, issue_raw,
                         sentiment, sentiment_score, evidence_text, debug_features)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    debug_features = {
                        "matched_seed": aspect_data.get("matched_seed"),
                        "sentiment_words": aspect_data.get("sentiment_words", []),
                        "extraction_pattern": aspect_data.get("extraction_pattern")
                    }
                    
                    self.db.execute_write(insert_query, {
                        "run_id": self.run_id,
                        "pipeline_version": self.pipeline_version,
                        "data_slice_id": self.data_slice_id,
                        "created_at": self.version_fields["created_at"],
                        "sentence_id": sentence_id,
                        "extract_method": self.extract_method,
                        "aspect_raw": aspect_data["aspect_raw"],
                        "issue_raw": issue_raw,
                        "sentiment": aspect_data["sentiment"],
                        "sentiment_score": aspect_data["sentiment_score"],
                        "evidence_text": aspect_data["evidence_text"],
                        "debug_features": json.dumps(debug_features, ensure_ascii=False)
                    })
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"处理sentence_id={sentence_id}时出错: {str(e)}")
                fail_count += 1
        
        logger.info(
            f"TraditionalAspectSentimentExtractorAgent完成: "
            f"success={success_count}, fail={fail_count}, "
            f"总aspect数量={total_aspect_count}"
        )
        
        return {
            "status": "success",
            "success_count": success_count,
            "fail_count": fail_count,
            "total_aspect_count": total_aspect_count,
            "table": table_manager.ASPECT_SENTIMENT_RAW_TRADITIONAL
        }
    
    def _extract_aspects_and_sentiment(
        self,
        target_sentence: str,
        context_text: str,
        product_title: str = None,
        features_short: str = None
    ) -> List[Dict[str, Any]]:
        """
        阶段A-C：抽取aspect候选、过滤排序、sentiment判别
        
        Returns:
            aspect列表，每个包含aspect_raw, sentiment, sentiment_score, evidence_text
        """
        aspects = []
        
        # 阶段A：候选Aspect生成
        aspect_candidates = self._generate_aspect_candidates(
            target_sentence, context_text, product_title, features_short
        )
        
        if not aspect_candidates:
            return aspects
        
        # 阶段B：过滤与排序
        filtered_aspects = self._filter_and_rank_aspects(
            aspect_candidates, target_sentence, product_title, features_short
        )
        
        # 阶段C：Sentiment判别
        for aspect_candidate in filtered_aspects[:self.max_candidates_per_sentence]:
            sentiment, sentiment_score = self._classify_sentiment(
                target_sentence, aspect_candidate["text"]
            )
            
            if sentiment == "neutral" and sentiment_score < 0.3:
                # 跳过中性且低置信度的aspect
                continue
            
            aspects.append({
                "aspect_raw": aspect_candidate["text"],
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "evidence_text": aspect_candidate.get("evidence", target_sentence[:100]),
                "matched_seed": aspect_candidate.get("matched_seed"),
                "sentiment_words": aspect_candidate.get("sentiment_words", []),
                "extraction_pattern": aspect_candidate.get("pattern")
            })
        
        return aspects
    
    def _generate_aspect_candidates(
        self,
        target_sentence: str,
        context_text: str,
        product_title: str = None,
        features_short: str = None
    ) -> List[Dict[str, Any]]:
        """阶段A：生成aspect候选"""
        candidates = []
        sentence_lower = target_sentence.lower()
        
        # 方法1：种子词表匹配
        for seed in self.aspect_seed_lexicon:
            if seed in sentence_lower:
                # 找到包含seed的短语
                pattern = rf'\b\w*\s*{re.escape(seed)}\s*\w*\b'
                matches = re.finditer(pattern, sentence_lower)
                for match in matches:
                    candidate_text = match.group().strip()
                    if len(candidate_text.split()) <= 3:  # 限制长度
                        candidates.append({
                            "text": candidate_text,
                            "source": "seed_lexicon",
                            "matched_seed": seed,
                            "confidence": 0.8
                        })
        
        # 方法2：名词短语提取（简单启发式：形容词+名词 或 名词+名词）
        # 模式：形容词 + 名词
        adj_noun_pattern = r'\b(?:good|bad|great|excellent|poor|nice|fine|big|small|large|tiny|fast|slow|long|short|heavy|light|cheap|expensive|durable|fragile|strong|weak|clear|bright|dark|loud|quiet|smooth|rough|soft|hard|comfortable|uncomfortable)\s+(\w+(?:\s+\w+){0,2})\b'
        matches = re.finditer(adj_noun_pattern, sentence_lower)
        for match in matches:
            noun_phrase = match.group(1).strip()
            if noun_phrase not in self.stop_nouns and len(noun_phrase.split()) <= 3:
                candidates.append({
                    "text": noun_phrase,
                    "source": "adj_noun_pattern",
                    "confidence": 0.6
                })
        
        # 方法3：上下文回指补偿（如果target_sentence多为代词）
        if any(pronoun in sentence_lower for pronoun in ["it", "this", "that", "they"]):
            # 从前一句提取名词短语
            if context_text:
                prev_sentence = context_text.split(".")[-2] if "." in context_text else context_text
                for seed in self.aspect_seed_lexicon:
                    if seed in prev_sentence.lower():
                        candidates.append({
                            "text": seed,
                            "source": "context_reference",
                            "matched_seed": seed,
                            "confidence": 0.5,
                            "evidence": target_sentence[:100]  # evidence仍来自target
                        })
        
        # 去重
        seen = set()
        unique_candidates = []
        for cand in candidates:
            if cand["text"] not in seen:
                seen.add(cand["text"])
                unique_candidates.append(cand)
        
        return unique_candidates
    
    def _filter_and_rank_aspects(
        self,
        candidates: List[Dict[str, Any]],
        target_sentence: str,
        product_title: str = None,
        features_short: str = None
    ) -> List[Dict[str, Any]]:
        """阶段B：过滤与排序"""
        filtered = []
        
        for cand in candidates:
            text = cand["text"].lower()
            
            # 过滤泛化词
            if text in self.stop_nouns:
                continue
            
            # 过滤过短或过长
            if len(text) < 2 or len(text.split()) > 4:
                continue
            
            # 元数据约束（如果启用）
            if self.use_meta_context:
                meta_text = ""
                if product_title:
                    meta_text += product_title.lower() + " "
                if features_short:
                    meta_text += features_short.lower()
                
                if meta_text and any(word in meta_text for word in text.split()):
                    cand["confidence"] = cand.get("confidence", 0.5) + 0.2  # 提升优先级
            
            # 计算与情感线索的距离（简单：检查附近是否有情感词）
            sentence_lower = target_sentence.lower()
            text_pos = sentence_lower.find(text)
            if text_pos >= 0:
                # 检查前后窗口内的情感词
                window_start = max(0, text_pos - 20)
                window_end = min(len(sentence_lower), text_pos + len(text) + 20)
                window = sentence_lower[window_start:window_end]
                
                sentiment_words = []
                for word in self.positive_words | self.negative_words:
                    if word in window:
                        sentiment_words.append(word)
                
                if sentiment_words:
                    cand["sentiment_words"] = sentiment_words
                    cand["confidence"] = cand.get("confidence", 0.5) + 0.1
            
            filtered.append(cand)
        
        # 按confidence排序
        filtered.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
        
        return filtered
    
    def _classify_sentiment(
        self,
        sentence: str,
        aspect: str
    ) -> Tuple[str, float]:
        """
        阶段C：Sentiment判别
        
        Returns:
            (sentiment, sentiment_score)
        """
        sentence_lower = sentence.lower()
        aspect_lower = aspect.lower()
        
        # 找到aspect在句子中的位置
        aspect_pos = sentence_lower.find(aspect_lower)
        if aspect_pos < 0:
            aspect_pos = 0
        
        # 提取aspect周围的窗口（前后各30个字符）
        window_start = max(0, aspect_pos - 30)
        window_end = min(len(sentence_lower), aspect_pos + len(aspect_lower) + 30)
        window = sentence_lower[window_start:window_end]
        
        # 计算情感分数
        positive_score = 0.0
        negative_score = 0.0
        
        # 检查正面词
        for word in self.positive_words:
            if word in window:
                positive_score += 1.0
        
        # 检查负面词
        for word in self.negative_words:
            if word in window:
                negative_score += 1.0
        
        # 检查否定词（反转情感）
        has_negation = any(neg in window for neg in self.negation_words)
        if has_negation:
            positive_score, negative_score = negative_score, positive_score
        
        # 检查程度词（加权）
        intensity_count = sum(1 for word in self.intensity_words if word in window)
        if intensity_count > 0:
            positive_score *= (1 + intensity_count * 0.2)
            negative_score *= (1 + intensity_count * 0.2)
        
        # 归一化分数
        total_score = positive_score + negative_score
        if total_score == 0:
            return ("neutral", 0.0)
        
        positive_ratio = positive_score / total_score
        
        if positive_ratio > 0.6:
            sentiment = "positive"
            sentiment_score = min(1.0, positive_ratio)
        elif positive_ratio < 0.4:
            sentiment = "negative"
            sentiment_score = min(1.0, 1 - positive_ratio)
        else:
            sentiment = "neutral"
            sentiment_score = 0.5
        
        return (sentiment, sentiment_score)
    
    def _extract_issue_phrase(
        self,
        sentence: str,
        aspect: str,
        evidence: str
    ) -> str:
        """
        阶段D：Issue短语抽取
        
        Returns:
            issue_raw字符串
        """
        sentence_lower = sentence.lower()
        aspect_lower = aspect.lower()
        
        # 找到aspect位置
        aspect_pos = sentence_lower.find(aspect_lower)
        if aspect_pos < 0:
            # 如果找不到aspect，尝试从evidence中提取
            return self._extract_issue_from_evidence(sentence, evidence)
        
        # 提取aspect周围的窗口
        window_start = max(0, aspect_pos - 15)
        window_end = min(len(sentence_lower), aspect_pos + len(aspect_lower) + 15)
        window = sentence_lower[window_start:window_end]
        
        # 模板1：aspect + is/are + adj
        pattern1 = rf'{re.escape(aspect_lower)}\s+(?:is|are|was|were)\s+(\w+(?:\s+\w+)?)'
        match = re.search(pattern1, window)
        if match:
            return match.group(1).strip()
        
        # 模板2：aspect + verb + obj
        pattern2 = rf'{re.escape(aspect_lower)}\s+(\w+)\s+(\w+)'
        match = re.search(pattern2, window)
        if match:
            return f"{match.group(1)} {match.group(2)}".strip()
        
        # 模板3：too + adj
        pattern3 = r'too\s+(\w+)'
        match = re.search(pattern3, window)
        if match:
            return f"too {match.group(1)}".strip()
        
        # 模板4：not + adj
        pattern4 = r'not\s+(\w+)'
        match = re.search(pattern4, window)
        if match:
            return f"not {match.group(1)}".strip()
        
        # 回退：提取情感词 + 最近谓词/形容词
        issue_words = []
        for word in self.positive_words | self.negative_words:
            if word in window:
                issue_words.append(word)
                break
        
        # 如果找到情感词，添加附近的形容词或动词
        if issue_words:
            words = window.split()
            for i, word in enumerate(words):
                if word in issue_words:
                    # 取前后各一个词
                    if i > 0:
                        issue_words.append(words[i-1])
                    if i < len(words) - 1:
                        issue_words.append(words[i+1])
                    break
        
        if issue_words:
            return " ".join(issue_words[:3]).strip()
        
        # 最终回退：返回evidence的前几个词
        return self._extract_issue_from_evidence(sentence, evidence)
    
    def _extract_issue_from_evidence(self, sentence: str, evidence: str) -> str:
        """从evidence中提取issue"""
        if not evidence:
            return ""
        
        # 提取evidence的前几个词（最多5个）
        words = evidence.lower().split()[:5]
        return " ".join(words).strip()
    
    def _extract_evidence_from_sentence(
        self,
        sentence: str,
        aspect: str
    ) -> str:
        """从sentence中提取evidence（当原始evidence无法定位时）"""
        sentence_lower = sentence.lower()
        aspect_lower = aspect.lower()
        
        aspect_pos = sentence_lower.find(aspect_lower)
        if aspect_pos < 0:
            return sentence[:100]  # 回退到前100个字符
        
        # 提取aspect及其周围的文本（前后各20个字符）
        window_start = max(0, aspect_pos - 20)
        window_end = min(len(sentence), aspect_pos + len(aspect) + 20)
        return sentence[window_start:window_end].strip()
    
    def _validate_evidence(self, evidence: str, target_sentence: str) -> bool:
        """验证evidence可定位性"""
        if not evidence or not target_sentence:
            return False
        return evidence.lower() in target_sentence.lower()



