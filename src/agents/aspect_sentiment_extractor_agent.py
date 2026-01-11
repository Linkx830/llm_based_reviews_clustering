# 新增或修改我时需要修改这个文件夹中的README.md文件
"""AspectSentimentExtractorAgent - LLM结构化抽取"""
from typing import Dict, Any, List
import json
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..models.llm_wrapper import LLMWrapper
from ..tools.structured_output_tool import StructuredOutputTool
from ..utils.logger import get_logger
from ..utils.retry import retry_with_backoff
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class AspectItem(BaseModel):
    """方面抽取项（LLM输出，不包含位置信息）"""
    aspect: str = Field(description="方面名称（名词短语）")
    sentiment: str = Field(description="情感：positive/negative/neutral")
    sentiment_score: float = Field(default=0.0, description="情感分数")
    issue: str = Field(description="具体问题描述")
    evidence_text: str = Field(description="证据文本")
    confidence: float = Field(default=0.5, description="置信度")


class LLMExtractionOutput(BaseModel):
    """LLM抽取输出结构（不包含sentence_id和位置信息）"""
    has_opinion: bool = Field(description="是否包含观点")
    aspects: List[AspectItem] = Field(default_factory=list, description="方面列表")
    language: str = Field(default="en", description="语言")
    extraction_warnings: List[str] = Field(default_factory=list, description="警告")


class ExtractionOutput(BaseModel):
    """完整抽取输出结构（包含后处理添加的字段）"""
    sentence_id: str = Field(description="句子ID")
    has_opinion: bool = Field(description="是否包含观点")
    aspects: List[Dict[str, Any]] = Field(default_factory=list, description="方面列表（包含位置信息）")
    language: str = Field(default="en", description="语言")
    extraction_warnings: List[str] = Field(default_factory=list, description="警告")


class AspectSentimentExtractorAgent(BaseAgent):
    """
    AspectSentimentExtractorAgent
    
    职责：
    - 使用LLM从句子中抽取方面和情感
    - 输出结构化结果
    
    输入表：review_sentences, opinion_candidates, meta_context
    输出表：aspect_sentiment_raw
    """
    
    def __init__(
        self,
        *args,
        llm_wrapper: LLMWrapper,
        prompt_template: str = None,
        prompt_version: str = "v1.0",
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.llm = llm_wrapper
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        # LLM输出不包含sentence_id和位置信息，使用LLMExtractionOutput
        self.output_tool = StructuredOutputTool(LLMExtractionOutput, max_retries)
        
        # 加载prompt模板
        if prompt_template is None:
            from ..utils.config_loader import load_prompt_template
            self.prompt_template:str = load_prompt_template("extraction", prompt_version)
        else:
            self.prompt_template:str = prompt_template
    
    def process(self) -> Dict[str, Any]:
        """
        执行抽取
        
        Returns:
            处理结果统计
        """
        logger.info(f"开始执行AspectSentimentExtractorAgent，run_id={self.run_id}")
        
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

        total_sentence: int = len(candidates)
        
        # 检查是否有重复的sentence_id
        sentence_ids = [row[0] for row in candidates]
        unique_sentence_ids = set(sentence_ids)
        if len(sentence_ids) != len(unique_sentence_ids):
            duplicate_count = len(sentence_ids) - len(unique_sentence_ids)
            logger.warning(
                f"发现 {duplicate_count} 个重复的sentence_id！"
                f"总记录数: {total_sentence}, 唯一sentence_id数: {len(unique_sentence_ids)}"
            )
            # 去重处理
            seen = set()
            unique_candidates = []
            for row in candidates:
                sentence_id = row[0]
                if sentence_id not in seen:
                    seen.add(sentence_id)
                    unique_candidates.append(row)
            candidates = unique_candidates
            total_sentence = len(candidates)
            logger.info(f"去重后候选句总数: {total_sentence}")

        logger.info(f"候选句总数: {total_sentence}")

        table_manager = TableManager(self.db)
        success_count = 0
        fail_count = 0
        sentence_count = 0
        total_aspect_count = 0  # 统计总的aspect数量
        
        for row in candidates:
            sentence_id, target_sentence, context_text, parent_asin, product_title, features_short = row
            sentence_count += 1
            logger.info(f"Review sentence({sentence_count} / {total_sentence}): {target_sentence}")
            
            # 构建prompt
            prompt = self._build_prompt(
                target_sentence, context_text, product_title, features_short
            )
            
            # 调用LLM（带重试）
            retry_count = 0
            parse_status = "FAIL"
            error_type = None
            llm_output = None
            
            @retry_with_backoff(
                max_retries=self.max_retries,
                initial_delay=1.0,
                exceptions=(Exception,)
            )
            def call_llm():
                response = self.llm.invoke_structured(prompt, LLMExtractionOutput)
                
                # 处理响应：Ollama 和 OpenAI 的响应格式可能不同
                if hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, str):
                    response_text = response
                else:
                    response_text = str(response)
                
                # 解析响应
                parsed, error = self.output_tool.parse(response_text)
                if not parsed:
                    raise ValueError(error or "解析失败")
                return parsed
            
            try:
                parsed = call_llm()
                
                # 统计aspect数量
                aspect_count = len(parsed.aspects) if parsed.aspects else 0
                total_aspect_count += aspect_count
                
                # 后处理：添加sentence_id和位置信息
                full_output = self._post_process_output(parsed, sentence_id, target_sentence)
                llm_output = json.dumps(full_output, ensure_ascii=False)
                parse_status = "SUCCESS"
                logger.info(f"抽取成功: sentence_id={sentence_id}, aspect数量={aspect_count}, LLM output: {llm_output}")
            except Exception as e:
                error_type = str(e)
                retry_count = self.max_retries
                logger.warning(f"sentence_id={sentence_id} 抽取失败: {error_type}")
            
            # 插入aspect_sentiment_raw
            insert_query = f"""
                INSERT INTO {table_manager.ASPECT_SENTIMENT_RAW}
                (run_id, pipeline_version, data_slice_id, created_at,
                 llm_model, prompt_version, sentence_id,
                 parse_status, retry_count, error_type, llm_output)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.db.execute_write(insert_query, {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id,
                "created_at": self.version_fields["created_at"],
                "llm_model": self.llm.model_name,
                "prompt_version": self.prompt_version,
                "sentence_id": sentence_id,
                "parse_status": parse_status,
                "retry_count": retry_count,
                "error_type": error_type,
                "llm_output": llm_output
            })
            
            if parse_status == "SUCCESS":
                success_count += 1
            else:
                fail_count += 1
        
        logger.info(
            f"AspectSentimentExtractorAgent完成: "
            f"success={success_count}, fail={fail_count}, "
            f"总aspect数量={total_aspect_count}"
        )
        
        return {
            "status": "success",
            "success_count": success_count,
            "fail_count": fail_count,
            "total_aspect_count": total_aspect_count,
            "table": table_manager.ASPECT_SENTIMENT_RAW
        }
    
    def _build_prompt(
        self,
        target_sentence: str,
        context_text: str,
        product_title: str = None,
        features_short: str = None
    ) -> str:
        """构建prompt"""
        prompt = self.prompt_template.format(
            target_sentence=target_sentence,
            context_text=context_text,
            product_title=product_title or "",
            features_short=features_short or ""
        )
        return prompt
    
    def _post_process_output(
        self,
        llm_output: LLMExtractionOutput,
        sentence_id: str,
        target_sentence: str
    ) -> Dict[str, Any]:
        """
        后处理LLM输出：添加sentence_id和位置信息
        
        Args:
            llm_output: LLM原始输出
            sentence_id: 句子ID
            target_sentence: 目标句子（用于计算位置）
        
        Returns:
            完整的输出字典
        """
        # 转换为基础字典
        output_dict = llm_output.model_dump()
        
        # 添加sentence_id
        output_dict["sentence_id"] = sentence_id
        
        # 为每个aspect添加位置信息
        processed_aspects = []
        for aspect in output_dict.get("aspects", []):
            aspect_dict = aspect if isinstance(aspect, dict) else aspect.model_dump()
            evidence_text = aspect_dict.get("evidence_text", "")
            
            # 计算evidence在target_sentence中的位置
            start_char, end_char = self._find_evidence_position(
                evidence_text, target_sentence
            )
            aspect_dict["evidence_start_char"] = start_char
            aspect_dict["evidence_end_char"] = end_char
            
            processed_aspects.append(aspect_dict)
        
        output_dict["aspects"] = processed_aspects
        
        return output_dict
    
    def _find_evidence_position(self, evidence_text: str, target_sentence: str) -> tuple[int, int]:
        """
        查找evidence在target_sentence中的位置
        
        Args:
            evidence_text: 证据文本
            target_sentence: 目标句子
        
        Returns:
            (start_char, end_char)
        """
        if not evidence_text or not target_sentence:
            return (-1, -1)
        
        # 尝试精确匹配
        evidence_lower = evidence_text.lower().strip()
        target_lower = target_sentence.lower()
        
        start_idx = target_lower.find(evidence_lower)
        if start_idx >= 0:
            end_idx = start_idx + len(evidence_text)
            return (start_idx, end_idx)
        
        # 如果精确匹配失败，尝试部分匹配（取前几个词）
        evidence_words = evidence_lower.split()
        if evidence_words:
            # 尝试匹配前3个词
            search_text = " ".join(evidence_words[:3])
            start_idx = target_lower.find(search_text)
            if start_idx >= 0:
                # 找到匹配后，尝试找到完整evidence的结束位置
                end_idx = start_idx
                for word in evidence_words:
                    word_start = target_lower.find(word, end_idx)
                    if word_start >= 0:
                        end_idx = word_start + len(word)
                    else:
                        break
                return (start_idx, end_idx)
        
        return (-1, -1)

