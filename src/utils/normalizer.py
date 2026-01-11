# 新增或修改我时需要修改这个文件夹中的README.md文件
"""文本归一化工具 - 同义词归一和噪声处理"""
from typing import Dict, Set, Optional, Tuple, List, Union, Any
from difflib import SequenceMatcher
from .config_loader import load_taxonomy_files
import numpy as np

try:
    from ..tools.embedding_tool import EmbeddingTool
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    EmbeddingTool = None

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    Field = None


# 定义 LLM 输出的 Pydantic schema（用于 aspect 归一化）
if PYDANTIC_AVAILABLE:
    class AspectNormalizationOutput(BaseModel):
        """LLM 输出的 aspect 归一化结果"""
        normalized_aspect: str = Field(description="归一化后的aspect，可以从词表中选择，也可以创建新的标准aspect")
        confidence: float = Field(description="置信度，0-1之间", ge=0.0, le=1.0)
        reasoning: Optional[str] = Field(default=None, description="选择该归一化aspect的理由")
        is_noise: bool = Field(default=False, description="是否为噪声词（如product/item/it等）")
        should_add_to_taxonomy: bool = Field(default=False, description="是否应该添加到词表中（当创建新aspect时设为true）")
        is_new_aspect: bool = Field(default=False, description="是否为新创建的aspect（不在原词表中）")
else:
    AspectNormalizationOutput = None


class TextNormalizer:
    """
    文本归一化器
    
    职责：
    - aspect同义词归一（支持多种匹配策略，包括LLM辅助匹配）
    - 噪声词识别
    - 文本清洗
    """
    
    def __init__(
        self, 
        taxonomy: Optional[Dict[str, Dict[str, str]]] = None,
        embedding_tool: Optional[EmbeddingTool] = None,
        use_semantic_matching: bool = False,
        semantic_threshold: float = 0.7,
        llm_wrapper=None,
        use_llm_matching: bool = False,
        llm_confidence_threshold: float = 0.5
    ):
        """
        Args:
            taxonomy: 词表字典（如果为None则从文件加载）
            embedding_tool: Embedding工具（用于语义相似度匹配，可选）
            use_semantic_matching: 是否启用语义相似度匹配（需要embedding_tool）
            semantic_threshold: 语义相似度阈值（默认0.7）
            llm_wrapper: LLM包装器（用于LLM辅助匹配，可选）
            use_llm_matching: 是否启用LLM辅助匹配（需要llm_wrapper）
            llm_confidence_threshold: LLM匹配的置信度阈值（默认0.5）
        """
        if taxonomy is None:
            taxonomy = load_taxonomy_files()
        
        self.aspect_synonyms = taxonomy.get("aspect_synonyms", {})
        self.noise_terms = set(taxonomy.get("noise_terms", {}).keys())
        self.aspect_allowlist = set(taxonomy.get("aspect_allowlist", {}).keys())
        
        # 构建归一化aspect集合（用于匹配）
        self.normalized_aspects = set(self.aspect_synonyms.values())
        
        # Embedding相关
        self.embedding_tool = embedding_tool
        self.use_semantic_matching = use_semantic_matching and embedding_tool is not None
        self.semantic_threshold = semantic_threshold
        self._normalized_embeddings_cache = None  # 缓存归一化aspect的embedding
        
        # LLM相关
        self.llm_wrapper = llm_wrapper
        self.use_llm_matching = use_llm_matching and llm_wrapper is not None
        self.llm_confidence_threshold = llm_confidence_threshold
        self._llm_match_cache = {}  # 缓存LLM匹配结果，避免重复调用
        self._new_aspects = set()  # 记录LLM新增的aspect
        self._aspect_synonyms_dynamic = {}  # 动态添加的同义词映射（aspect -> normalized_aspect）
        
        # 默认噪声词（如果词表为空）
        if not self.noise_terms:
            self.noise_terms = {
                "product", "item", "thing", "it", "quality", "review",
                "stuff", "this", "that", "one"
            }
    
    def normalize_aspect(
        self, 
        aspect: str, 
        return_match_info: bool = False
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        归一化aspect（增强版，支持多种匹配策略）
        
        匹配策略（按优先级）：
        1. 精确匹配：完全匹配同义词表
        2. 部分匹配：aspect包含同义词表中的词
        3. 模糊匹配：使用编辑距离（SequenceMatcher）
        4. 关键词提取匹配：提取核心词进行匹配
        5. 语义相似度匹配：使用embedding计算相似度（如果启用）
        6. LLM辅助匹配：使用大模型判断最合适的归一化aspect（如果启用）
        
        Args:
            aspect: 原始aspect
            return_match_info: 是否返回匹配信息（用于调试）
        
        Returns:
            归一化后的aspect，如果return_match_info=True则返回(aspect, match_info)
        """
        if not aspect:
            result = ""
            if return_match_info:
                return result, {"method": "empty", "confidence": 0.0}
            return result
        
        aspect_lower = aspect.lower().strip()
        match_info = {"original": aspect, "normalized": aspect_lower, "method": "none", "confidence": 0.0}
        
        # 策略1: 精确匹配
        if aspect_lower in self.aspect_synonyms:
            result = self.aspect_synonyms[aspect_lower]
            match_info.update({"method": "exact", "confidence": 1.0})
            if return_match_info:
                return result, match_info
            return result
        
        # 策略2: 部分匹配（检查是否包含同义词）
        best_partial_match = None
        best_partial_score = 0.0
        for original, normalized in self.aspect_synonyms.items():
            if original in aspect_lower:
                # 计算匹配度：匹配长度 / aspect总长度
                match_score = len(original) / len(aspect_lower) if aspect_lower else 0
                if match_score > best_partial_score:
                    best_partial_score = match_score
                    best_partial_match = normalized
        
        if best_partial_match and best_partial_score > 0.5:  # 至少匹配50%
            match_info.update({
                "method": "partial", 
                "confidence": best_partial_score,
                "matched_term": best_partial_match
            })
            if return_match_info:
                return best_partial_match, match_info
            return best_partial_match
        
        # 策略3: 模糊匹配（使用SequenceMatcher）
        best_fuzzy_match = None
        best_fuzzy_score = 0.0
        fuzzy_threshold = 0.75  # 模糊匹配阈值
        
        for original, normalized in self.aspect_synonyms.items():
            similarity = SequenceMatcher(None, aspect_lower, original).ratio()
            if similarity > best_fuzzy_score and similarity >= fuzzy_threshold:
                best_fuzzy_score = similarity
                best_fuzzy_match = normalized
        
        if best_fuzzy_match:
            match_info.update({
                "method": "fuzzy", 
                "confidence": best_fuzzy_score,
                "matched_term": best_fuzzy_match
            })
            if return_match_info:
                return best_fuzzy_match, match_info
            return best_fuzzy_match
        
        # 策略4: 关键词提取匹配
        aspect_words = set(aspect_lower.split())
        best_keyword_match = None
        best_keyword_score = 0.0
        
        for original, normalized in self.aspect_synonyms.items():
            original_words = set(original.split())
            # 计算词重叠度
            overlap = len(aspect_words & original_words)
            if overlap > 0:
                # Jaccard相似度
                union = len(aspect_words | original_words)
                jaccard = overlap / union if union > 0 else 0
                if jaccard > best_keyword_score and jaccard >= 0.5:  # 至少50%重叠
                    best_keyword_score = jaccard
                    best_keyword_match = normalized
        
        if best_keyword_match:
            match_info.update({
                "method": "keyword", 
                "confidence": best_keyword_score,
                "matched_term": best_keyword_match
            })
            if return_match_info:
                return best_keyword_match, match_info
            return best_keyword_match
        
        # 策略5: 语义相似度匹配（如果启用）
        if self.use_semantic_matching:
            semantic_match = self._semantic_match(aspect_lower)
            if semantic_match:
                matched_term, similarity = semantic_match
                match_info.update({
                    "method": "semantic", 
                    "confidence": similarity,
                    "matched_term": matched_term
                })
                if return_match_info:
                    return matched_term, match_info
                return matched_term
        
        # 策略6: LLM辅助匹配（如果启用，作为最后手段）
        if self.use_llm_matching:
            llm_match = self._llm_match(aspect_lower)
            if llm_match:
                matched_term, confidence, is_noise = llm_match
                if not is_noise and confidence >= self.llm_confidence_threshold:
                    match_info.update({
                        "method": "llm", 
                        "confidence": confidence,
                        "matched_term": matched_term
                    })
                    if return_match_info:
                        return matched_term, match_info
                    return matched_term
                elif is_noise:
                    # LLM判断为噪声，返回原始值但标记为噪声
                    match_info.update({
                        "method": "llm_noise", 
                        "confidence": confidence,
                        "is_noise": True
                    })
                    if return_match_info:
                        return aspect_lower, match_info
                    return aspect_lower
        
        # 如果所有策略都失败，返回原始值（小写）
        match_info.update({"method": "fallback", "confidence": 0.0})
        if return_match_info:
            return aspect_lower, match_info
        return aspect_lower
    
    def _semantic_match(self, aspect: str) -> Optional[Tuple[str, float]]:
        """
        使用embedding进行语义相似度匹配
        
        Args:
            aspect: 待匹配的aspect
        
        Returns:
            (matched_term, similarity) 或 None
        """
        if not self.embedding_tool or not self.normalized_aspects:
            return None
        
        try:
            # 缓存归一化aspect的embedding（首次调用时计算）
            if self._normalized_embeddings_cache is None:
                normalized_list = list(self.normalized_aspects)
                self._normalized_embeddings_cache = {
                    "aspects": normalized_list,
                    "embeddings": self.embedding_tool.encode(normalized_list)
                }
            
            # 计算aspect的embedding
            aspect_embedding = self.embedding_tool.encode_single(aspect)
            
            # 计算与所有归一化aspect的相似度
            best_match = None
            best_similarity = 0.0
            
            for i, normalized_aspect in enumerate(self._normalized_embeddings_cache["aspects"]):
                normalized_embedding = self._normalized_embeddings_cache["embeddings"][i]
                # 计算余弦相似度
                similarity = self._cosine_similarity(aspect_embedding, normalized_embedding)
                if similarity > best_similarity and similarity >= self.semantic_threshold:
                    best_similarity = similarity
                    best_match = normalized_aspect
            
            if best_match:
                return (best_match, best_similarity)
        except Exception as e:
            # 如果embedding失败，静默返回None（不影响其他策略）
            from ..utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"语义匹配失败: {e}")
        
        return None
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _llm_match(self, aspect: str) -> Optional[Tuple[str, float, bool]]:
        """
        使用LLM进行aspect归一化匹配
        
        Args:
            aspect: 待匹配的aspect
        
        Returns:
            (matched_term, confidence, is_noise) 或 None
        """
        if not self.llm_wrapper or not self.normalized_aspects or not PYDANTIC_AVAILABLE:
            return None
        
        # 检查缓存
        if aspect in self._llm_match_cache:
            return self._llm_match_cache[aspect]
        
        try:
            # 构建prompt
            normalized_list = sorted(list(self.normalized_aspects))
            prompt = self._build_llm_normalization_prompt(aspect, normalized_list)
            
            # 调用LLM
            response = self.llm_wrapper.invoke_structured(prompt, AspectNormalizationOutput)
            
            # 解析响应
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # 使用 StructuredOutputTool 解析
            from ..tools.structured_output_tool import StructuredOutputTool
            output_tool = StructuredOutputTool(AspectNormalizationOutput, max_retries=1)
            parsed, error = output_tool.parse(response_text)
            
            if not parsed:
                # 如果解析失败，尝试直接解析JSON
                import json
                try:
                    json_str = output_tool._extract_json(response_text)
                    data = json.loads(json_str)
                    parsed = AspectNormalizationOutput(**data)
                except Exception as e:
                    from ..utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.warning(f"LLM匹配解析失败: {e}, aspect={aspect}")
                    return None
            
            # 处理归一化aspect
            normalized_aspect = parsed.normalized_aspect.lower().strip()
            aspect_lower = aspect.lower().strip()
            
            # 如果LLM判断应该添加到词表且是新aspect
            if parsed.should_add_to_taxonomy and parsed.is_new_aspect:
                # 将新aspect添加到动态词表
                if normalized_aspect not in self.normalized_aspects:
                    self.normalized_aspects.add(normalized_aspect)
                    self._new_aspects.add(normalized_aspect)
                    # 建立同义词映射
                    self._aspect_synonyms_dynamic[aspect_lower] = normalized_aspect
                    self.aspect_synonyms[aspect_lower] = normalized_aspect
                    
                    from ..utils.logger import get_logger
                    logger = get_logger(__name__)
                    logger.info(
                        f"LLM新增aspect到词表: {aspect} -> {normalized_aspect} "
                        f"(confidence={parsed.confidence:.2f}, reasoning={parsed.reasoning or 'N/A'})"
                    )
            
            # 如果normalized_aspect不在词表中（且不是新aspect），记录警告但使用它
            if normalized_aspect not in self.normalized_aspects:
                from ..utils.logger import get_logger
                logger = get_logger(__name__)
                if not parsed.is_new_aspect:
                    logger.warning(
                        f"LLM返回的归一化aspect不在词表中: {normalized_aspect}, "
                        f"原始aspect={aspect}, 但LLM未标记为新aspect，使用LLM返回的值"
                    )
                # 即使不在词表中，也使用LLM返回的值（可能是新aspect）
                # 同时添加到动态词表
                if normalized_aspect not in self._new_aspects:
                    self.normalized_aspects.add(normalized_aspect)
                    self._new_aspects.add(normalized_aspect)
                    self._aspect_synonyms_dynamic[aspect_lower] = normalized_aspect
                    self.aspect_synonyms[aspect_lower] = normalized_aspect
            
            result = (normalized_aspect, parsed.confidence, parsed.is_noise)
            
            # 缓存结果
            self._llm_match_cache[aspect] = result
            return result
            
        except Exception as e:
            # 如果LLM调用失败，静默返回None（不影响其他策略）
            from ..utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"LLM匹配失败: {e}, aspect={aspect}")
            return None
    
    def _build_llm_normalization_prompt(self, aspect: str, normalized_aspects: List[str]) -> str:
        """
        构建LLM归一化prompt
        
        Args:
            aspect: 待归一化的aspect
            normalized_aspects: 归一化aspect列表
        
        Returns:
            prompt文本
        """
        aspects_str = ", ".join(sorted(normalized_aspects))
        
        prompt = f"""你是一个文本归一化专家。请将给定的aspect归一化到标准词表中，如果词表不足，你有权创建新的标准aspect。

            **任务**：
            将以下aspect归一化到标准词表中的最合适的aspect。如果aspect是噪声词（如product/item/it/thing等），请标记为噪声。如果词表中没有合适的aspect，你可以创建新的标准aspect。

            **待归一化的aspect**：{aspect}

            **标准词表**（优先从以下选择，如果都不合适可以创建新的）：
            {aspects_str}

            **要求**：
            1. **优先匹配**：如果aspect在词表中或与词表中的某个aspect语义相同/相似，选择最匹配的归一化aspect，设置is_new_aspect=false
            2. **噪声识别**：如果aspect是噪声词（过于泛化，如product/item/it/thing/quality等），设置is_noise=true，normalized_aspect可以是aspect本身或最接近的词
            3. **创建新aspect**：如果aspect不在词表中，且不是噪声词，且是一个有意义的方面（如"充电速度"、"屏幕亮度"等），你可以：
            - 创建一个新的标准aspect名称（使用简洁、通用的名词短语，如"charging speed"、"screen brightness"）
            - 设置normalized_aspect为你创建的新aspect名称
            - 设置is_new_aspect=true
            - 设置should_add_to_taxonomy=true
            - 在reasoning中说明为什么创建这个新aspect
            4. **新aspect命名规范**：
            - 使用简洁的名词短语（1-3个词）
            - 使用英文（如果原aspect是中文，翻译成英文）
            - 使用小写
            - 避免过于具体或过于泛化
            - 示例：如果aspect是"充电很快"，可以创建"charging speed"；如果是"屏幕很亮"，可以创建"screen brightness"

            **输出格式**（JSON）：
            {{
                "normalized_aspect": "归一化后的aspect（来自词表或新创建）",
                "confidence": 0.0-1.0之间的置信度,
                "reasoning": "选择或创建理由（必填，如果是新aspect请说明）",
                "is_noise": false,
                "should_add_to_taxonomy": true/false（如果是新aspect必须为true）,
                "is_new_aspect": true/false（如果是新创建的aspect设为true）
            }}

            请直接输出JSON，不要包含其他文本。
        """
        
        return prompt
    
    def normalize_issue(self, issue: str) -> str:
        """
        归一化issue
        
        Args:
            issue: 原始issue
        
        Returns:
            归一化后的issue
        """
        if not issue:
            return ""
        
        # 简单清洗：转小写、去前后空白
        return issue.lower().strip()
    
    def is_noise_aspect(self, aspect: str) -> bool:
        """
        判断aspect是否为噪声
        
        Args:
            aspect: aspect文本
        
        Returns:
            True if 噪声
        """
        if not aspect:
            return True
        
        aspect_lower = aspect.lower().strip()
        
        # 检查噪声词表
        if aspect_lower in self.noise_terms:
            return True
        
        # 检查是否完全由噪声词组成
        words = aspect_lower.split()
        if all(word in self.noise_terms for word in words):
            return True
        
        # 检查是否在允许列表中
        if aspect_lower in self.aspect_allowlist:
            return False
        
        return False
    
    def is_noise_issue(self, issue: str) -> bool:
        """
        判断issue是否为噪声
        
        Args:
            issue: issue文本
        
        Returns:
            True if 噪声
        """
        if not issue:
            return True
        
        issue_lower = issue.lower().strip()
        
        # 噪声issue模式
        noise_patterns = {
            "good", "bad", "nice", "works", "not good", "not bad",
            "ok", "okay", "fine", "great", "terrible", "awful"
        }
        
        if issue_lower in noise_patterns:
            return True
        
        # 检查是否过短且无修饰
        if len(issue_lower.split()) <= 1 and issue_lower in noise_patterns:
            return True
        
        return False
    
    def judge_validity(self, aspect: str, issue: str) -> str:
        """
        判断有效性
        
        Args:
            aspect: aspect文本
            issue: issue文本
        
        Returns:
            "VALID", "NOISE", 或 "INVALID"
        """
        if not aspect or not issue:
            return "INVALID"
        
        if self.is_noise_aspect(aspect):
            return "NOISE"
        
        if self.is_noise_issue(issue):
            return "NOISE"
        
        return "VALID"
    
    def get_new_aspects(self) -> Set[str]:
        """
        获取LLM新增的aspect集合
        
        Returns:
            新增的aspect集合
        """
        return self._new_aspects.copy()
    
    def get_dynamic_synonyms(self) -> Dict[str, str]:
        """
        获取动态添加的同义词映射
        
        Returns:
            动态同义词映射字典
        """
        return self._aspect_synonyms_dynamic.copy()

