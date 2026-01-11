# 新增或修改我时需要修改这个文件夹中的README.md文件
"""ClusterInsightAgent - 簇命名、摘要、建议、优先级"""
from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, Field
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..models.llm_wrapper import LLMWrapper
from ..tools.structured_output_tool import StructuredOutputTool
from ..utils.logger import get_logger
from ..utils.retry import retry_with_backoff

logger = get_logger(__name__)


class EvidenceItem(BaseModel):
    """证据条目"""
    sentence_id: str = Field(description="句子ID，必须来自输入样本")
    quote: str = Field(description="引用文本，必须与对应样本的target_sentence一致或为其子串")
    why_representative: Optional[str] = Field(default=None, description="为何典型的说明")


class ActionItem(BaseModel):
    """可执行建议条目"""
    action: str = Field(description="建议动作本体（具体、可落地）")
    owner_team: Optional[str] = Field(default=None, description="建议责任方：product/design/quality/logistics/customer_service等")
    expected_impact: Optional[str] = Field(default=None, description="预期影响（如降低退货率、提升满意度）")
    validation_plan: Optional[str] = Field(default=None, description="如何验证有效（如A/B、抽检、实验）")
    urgency: Optional[str] = Field(default=None, description="紧急程度：high/medium/low")


class ClusterInsightOutput(BaseModel):
    """簇洞察输出结构（LLM输出）"""
    cluster_name: str = Field(description="簇名称，一句话命名，必须体现aspect，建议≤12词（英文）或≤20字（中文）")
    summary: str = Field(description="2-3句现象总结：描述发生了什么+对用户影响")
    priority: str = Field(description="优先级：high/medium/low")
    priority_rationale: Optional[str] = Field(default=None, description="简短说明优先级依据（规模/负面率/趋势）")
    evidence_items: List[EvidenceItem] = Field(description="证据条目列表，必须可回溯到输入样本")
    action_items: List[ActionItem] = Field(description="可执行建议条目列表，建议3条")
    risks_and_assumptions: Optional[List[str]] = Field(default=None, description="风险/假设列表，避免过度确定性")
    confidence: Optional[float] = Field(default=None, description="置信度，0~1，表示洞察可靠程度")


class ClusterInsightAgent(BaseAgent):
    """
    ClusterInsightAgent
    
    职责：
    - 对每个簇生成可读可执行的洞察条目
    - 簇命名、摘要、证据、建议、优先级
    
    输入表：cluster_stats, aspect_sentiment_valid
    输出表：cluster_reports
    """
    
    def __init__(
        self,
        *args,
        llm_wrapper: LLMWrapper,
        prompt_template: str = None,
        prompt_version: str = "v1.0",
        max_retries: int = 3,
        embedding_tool = None,  # 可选的embedding工具，用于向量相似度匹配
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.llm = llm_wrapper
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.embedding_tool = embedding_tool
        self.output_tool = StructuredOutputTool(ClusterInsightOutput, max_retries)
        
        # 加载prompt模板
        if prompt_template is None:
            from ..utils.config_loader import load_prompt_template
            self.prompt_template = load_prompt_template("insight", prompt_version)
        else:
            self.prompt_template = prompt_template
    
    def process(self) -> Dict[str, Any]:
        """生成簇洞察"""
        logger.info(f"开始执行ClusterInsightAgent，run_id={self.run_id}")
        
        # 读取cluster_stats
        query = f"""
            SELECT aspect_norm, cluster_id, cluster_size, neg_ratio,
                   representative_sentence_ids
            FROM {TableManager.CLUSTER_STATS}
            WHERE run_id = ?
            ORDER BY cluster_size DESC
            LIMIT 50
        """
        clusters = self.db.execute_read(query, {"run_id": self.run_id})
        
        logger.info(f"读取到 {len(clusters)} 个簇，开始生成洞察")
        
        table_manager = TableManager(self.db)
        processed_count = 0
        success_count = 0
        fail_count = 0
        
        for idx, row in enumerate(clusters, 1):
            aspect_norm, cluster_id, cluster_size, neg_ratio, rep_sentence_ids_json = row
            
            logger.info(
                f"处理簇 {idx}/{len(clusters)}: aspect={aspect_norm}, "
                f"cluster_id={cluster_id}, size={cluster_size}"
            )
            
            # 获取代表样本
            rep_sentence_ids = json.loads(rep_sentence_ids_json) if rep_sentence_ids_json else []
            
            # 获取样本详情
            samples = []
            if rep_sentence_ids:
                placeholders = ",".join(["?"] * len(rep_sentence_ids))
                sample_query = f"""
                    SELECT rs.sentence_id, rs.target_sentence, asv.issue_norm, asv.sentiment,
                           rs.helpful_vote, rs.timestamp
                    FROM {TableManager.REVIEW_SENTENCES} rs
                    JOIN {TableManager.ASPECT_SENTIMENT_VALID} asv
                        ON rs.sentence_id = asv.sentence_id AND rs.run_id = asv.run_id
                    WHERE rs.sentence_id IN ({placeholders}) AND rs.run_id = ?
                    LIMIT 20
                """
                params = list(rep_sentence_ids) + [self.run_id]
                samples = self.db.execute_read(sample_query, params)
            
            # 构建prompt并调用LLM生成洞察
            prompt = self._build_prompt(
                aspect_norm, cluster_id, cluster_size, neg_ratio, samples
            )
            
            # 调用LLM生成洞察
            insight = None
            try:
                insight = self._generate_insight_with_llm(prompt)
                success_count += 1
            except Exception as e:
                logger.warning(
                    f"簇 {cluster_id} (aspect={aspect_norm}) LLM生成失败: {e}，"
                    f"使用简化版本"
                )
                # 失败时使用简化版本作为fallback
                insight = self._generate_insight_simple(
                    aspect_norm, cluster_id, cluster_size, neg_ratio, samples
                )
                fail_count += 1
            
            # 验证证据可回溯性
            insight = self._validate_evidence_items(insight, samples)
            
            # 插入cluster_reports
            insert_query = f"""
                INSERT INTO {table_manager.CLUSTER_REPORTS}
                (run_id, pipeline_version, data_slice_id, created_at,
                 llm_model, prompt_version,
                 aspect_norm, cluster_id, cluster_name, summary,
                 priority, evidence_items, action_items, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # 转换evidence_items和action_items为JSON
            evidence_json = json.dumps(
                [item.model_dump() if isinstance(item, EvidenceItem) else item 
                 for item in insight.get("evidence_items", [])],
                ensure_ascii=False
            )
            action_json = json.dumps(
                [item.model_dump() if isinstance(item, ActionItem) else item 
                 for item in insight.get("action_items", [])],
                ensure_ascii=False
            )
            
            # 处理confidence字段，确保有默认值
            llm_confidence = insight.get("confidence")
            if llm_confidence is None:
                # 如果没有confidence，根据cluster_size和neg_ratio估算
                if cluster_size > 5:
                    llm_confidence = 0.8  # 大簇默认较高置信度
                elif cluster_size > 2:
                    llm_confidence = 0.7  # 中等簇
                else:
                    llm_confidence = 0.6  # 小簇默认较低置信度
                logger.debug(f"簇 {cluster_id} LLM未返回confidence，使用默认值: {llm_confidence}")
            
            self.db.execute_write(insert_query, {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id,
                "created_at": self.version_fields["created_at"],
                "llm_model": self.llm.model_name,
                "prompt_version": self.prompt_version,
                "aspect_norm": aspect_norm,
                "cluster_id": str(cluster_id),
                "cluster_name": insight.get("cluster_name", ""),
                "summary": insight.get("summary", ""),
                "priority": insight.get("priority", "medium"),
                "evidence_items": evidence_json,
                "action_items": action_json,
                "confidence": llm_confidence
            })
            processed_count += 1
        
        logger.info(
            f"ClusterInsightAgent完成: processed={processed_count}, "
            f"success={success_count}, fail={fail_count}"
        )
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "success_count": success_count,
            "fail_count": fail_count,
            "table": table_manager.CLUSTER_REPORTS
        }
    
    def _build_prompt(
        self,
        aspect_norm: str,
        cluster_id: str,
        cluster_size: int,
        neg_ratio: float,
        samples: List
    ) -> str:
        """构建prompt"""
        # 格式化样本数据
        samples_text = self._format_samples(samples)
        
        return self.prompt_template.format(
            aspect_norm=aspect_norm,
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            neg_ratio=neg_ratio,
            samples=samples_text
        )
    
    def _format_samples(self, samples: List) -> str:
        """格式化样本数据为文本"""
        if not samples:
            return "无代表样本"
        
        formatted = []
        for idx, sample in enumerate(samples[:20], 1):  # 最多20个样本
            sentence_id = sample[0]
            target_sentence = sample[1]
            issue_norm = sample[2] if len(sample) > 2 else ""
            sentiment = sample[3] if len(sample) > 3 else ""
            helpful_vote = sample[4] if len(sample) > 4 else 0
            timestamp = sample[5] if len(sample) > 5 else 0
            
            formatted.append(
                f"样本{idx}:\n"
                f"  - sentence_id: {sentence_id}\n"
                f"  - 句子: {target_sentence}\n"
                f"  - 问题: {issue_norm}\n"
                f"  - 情感: {sentiment}\n"
                f"  - 有用票数: {helpful_vote}\n"
            )
        
        return "\n".join(formatted)
    
    def _generate_insight_with_llm(self, prompt: str) -> Dict[str, Any]:
        """使用LLM生成洞察"""
        @retry_with_backoff(
            max_retries=self.max_retries,
            initial_delay=1.0,
            exceptions=(Exception,)
        )
        def call_llm():
            response = self.llm.invoke_structured(prompt, ClusterInsightOutput)
            
            # 如果响应已经是Pydantic模型，直接使用
            if isinstance(response, ClusterInsightOutput):
                logger.debug("LLM返回了已解析的Pydantic模型")
                return response
            
            # 处理响应：Ollama 和 OpenAI 的响应格式可能不同
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # 先尝试修复格式问题（在解析之前）
            fixed_data = self._fix_llm_output_format(response_text)
            if fixed_data:
                # 使用修复后的数据尝试创建模型
                try:
                    parsed = ClusterInsightOutput(**fixed_data)
                    logger.debug("使用修复后的数据成功创建模型")
                    return parsed
                except Exception as fix_error:
                    logger.warning(f"修复后的数据仍然无法通过验证: {fix_error}")
                    # 继续尝试原始解析
            
            # 尝试原始解析
            parsed, error = self.output_tool.parse(response_text)
            if not parsed:
                raise ValueError(error or "解析失败")
            return parsed
        
        try:
            parsed = call_llm()
            # 转换为字典格式
            insight_dict = parsed.model_dump()
            return insight_dict
        except Exception as e:
            logger.error(f"LLM生成洞察失败: {e}")
            raise
    
    def _fix_llm_output_format(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        尝试修复LLM输出格式问题
        
        处理情况：
        1. evidence_items 和 action_items 是字符串数组，需要转换为对象数组
        2. 其他格式问题
        """
        try:
            # 提取JSON
            json_str = self.output_tool._extract_json(response_text)
            data = json.loads(json_str)
            
            logger.debug(f"提取的JSON数据: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
            
            # 检查是否需要修复
            needs_fix = False
            
            # 修复 evidence_items
            if "evidence_items" in data and isinstance(data["evidence_items"], list):
                fixed_evidence = []
                for item in data["evidence_items"]:
                    if isinstance(item, str):
                        # 字符串格式，转换为对象
                        # 注意：sentence_id先设为空字符串，后续通过_validate_evidence_items匹配填充
                        needs_fix = True
                        fixed_evidence.append({
                            "sentence_id": "",  # 必需字段，先设为空，后续通过匹配填充
                            "quote": item,  # 必需字段
                            "why_representative": None  # 可选字段
                        })
                    elif isinstance(item, dict):
                        # 已经是对象，但确保必需字段存在
                        fixed_item = {
                            "sentence_id": item.get("sentence_id", ""),  # 必需字段，如果缺失则设为空字符串
                            "quote": item.get("quote", item.get("text", item.get("evidence", ""))),  # 必需字段
                            "why_representative": item.get("why_representative")  # 可选字段
                        }
                        # 确保quote不为空
                        if not fixed_item["quote"]:
                            fixed_item["quote"] = str(item.get("text", item.get("evidence", "")))
                        fixed_evidence.append(fixed_item)
                    else:
                        # 跳过无效项
                        continue
                data["evidence_items"] = fixed_evidence
                if needs_fix:
                    logger.info(f"修复了 {len(fixed_evidence)} 个evidence_items（从字符串转换为对象）")
                    logger.debug(f"修复后的evidence_items: {json.dumps(fixed_evidence[:2], ensure_ascii=False, indent=2)}")
            
            # 修复 action_items
            needs_fix = False
            if "action_items" in data and isinstance(data["action_items"], list):
                fixed_actions = []
                for item in data["action_items"]:
                    if isinstance(item, str):
                        # 字符串格式，转换为对象
                        needs_fix = True
                        fixed_actions.append({
                            "action": item,  # 必需字段
                            "owner_team": None,  # 可选字段
                            "expected_impact": None,  # 可选字段
                            "validation_plan": None,  # 可选字段
                            "urgency": None  # 可选字段
                        })
                    elif isinstance(item, dict):
                        # 已经是对象，但确保必需字段存在
                        fixed_item = {
                            "action": item.get("action", item.get("text", item.get("suggestion", ""))),  # 必需字段
                            "owner_team": item.get("owner_team"),  # 可选字段
                            "expected_impact": item.get("expected_impact"),  # 可选字段
                            "validation_plan": item.get("validation_plan"),  # 可选字段
                            "urgency": item.get("urgency")  # 可选字段
                        }
                        # 确保action不为空
                        if not fixed_item["action"]:
                            fixed_item["action"] = str(item.get("text", item.get("suggestion", "")))
                        fixed_actions.append(fixed_item)
                    else:
                        # 跳过无效项
                        continue
                data["action_items"] = fixed_actions
                if needs_fix:
                    logger.info(f"修复了 {len(fixed_actions)} 个action_items（从字符串转换为对象）")
                    logger.debug(f"修复后的action_items: {json.dumps(fixed_actions[:2], ensure_ascii=False, indent=2)}")
            
            # 确保其他必需字段存在
            if "cluster_name" not in data or not data["cluster_name"]:
                data["cluster_name"] = "未命名簇"
            if "summary" not in data or not data["summary"]:
                data["summary"] = "无摘要"
            if "priority" not in data or data["priority"] not in ["high", "medium", "low"]:
                data["priority"] = "medium"
            
            # 确保列表字段存在
            if "evidence_items" not in data:
                data["evidence_items"] = []
            if "action_items" not in data:
                data["action_items"] = []
            
            # 最终验证：确保所有evidence_items都有必需的字段
            for idx, item in enumerate(data.get("evidence_items", [])):
                if isinstance(item, dict):
                    if "sentence_id" not in item:
                        item["sentence_id"] = ""
                    if "quote" not in item or not item["quote"]:
                        logger.warning(f"evidence_items[{idx}]缺少quote字段，跳过")
                        data["evidence_items"][idx] = None
                elif not isinstance(item, str):
                    logger.warning(f"evidence_items[{idx}]格式不正确: {type(item)}")
                    data["evidence_items"][idx] = None
            
            # 移除None项
            data["evidence_items"] = [item for item in data.get("evidence_items", []) if item is not None]
            
            # 最终验证：确保所有action_items都有必需的字段
            for idx, item in enumerate(data.get("action_items", [])):
                if isinstance(item, dict):
                    if "action" not in item or not item["action"]:
                        logger.warning(f"action_items[{idx}]缺少action字段，跳过")
                        data["action_items"][idx] = None
                elif not isinstance(item, str):
                    logger.warning(f"action_items[{idx}]格式不正确: {type(item)}")
                    data["action_items"][idx] = None
            
            # 移除None项
            data["action_items"] = [item for item in data.get("action_items", []) if item is not None]
            
            logger.debug(f"修复完成，最终数据: evidence_items={len(data.get('evidence_items', []))}, action_items={len(data.get('action_items', []))}")
            
            return data
        except json.JSONDecodeError as e:
            logger.debug(f"无法解析JSON: {e}")
            return None
        except Exception as e:
            logger.debug(f"修复格式时出错: {e}")
            return None
    
    def _validate_evidence_items(
        self, 
        insight: Dict[str, Any], 
        samples: List
    ) -> Dict[str, Any]:
        """
        验证证据条目的可回溯性
        
        确保所有evidence_items中的sentence_id都来自输入样本
        如果sentence_id为空，尝试通过quote文本在样本中匹配
        """
        if not samples:
            return insight
        
        # 构建样本sentence_id集合和文本映射
        sample_ids = {s[0] for s in samples}
        sample_text_map = {s[0]: s[1] for s in samples}  # sentence_id -> target_sentence
        
        # 验证evidence_items
        evidence_items = insight.get("evidence_items", [])
        validated_evidence = []
        
        for evidence in evidence_items:
            if isinstance(evidence, dict):
                sentence_id = evidence.get("sentence_id", "")
                quote = evidence.get("quote", "")
                
                # 如果sentence_id为空，尝试通过quote文本匹配
                if not sentence_id and quote:
                    # 在样本中查找包含quote的句子
                    matched_id = None
                    quote_lower = quote.lower().strip()
                    for sid, text in sample_text_map.items():
                        if quote_lower in text.lower() or text.lower() in quote_lower:
                            matched_id = sid
                            break
                    
                    if matched_id:
                        sentence_id = matched_id
                        evidence["sentence_id"] = matched_id
                        logger.debug(f"通过quote文本匹配到sentence_id: {matched_id}")
                
                # 验证sentence_id是否在样本中
                if sentence_id and sentence_id in sample_ids:
                    validated_evidence.append(evidence)
                elif sentence_id:
                    logger.warning(
                        f"证据条目中的sentence_id={sentence_id}不在输入样本中，已过滤"
                    )
                elif quote:
                    # sentence_id为空且无法匹配，但保留quote作为证据
                    # 使用第一个样本的sentence_id作为占位符
                    if samples:
                        evidence["sentence_id"] = samples[0][0]
                        validated_evidence.append(evidence)
                        logger.debug(f"无法匹配sentence_id，使用第一个样本作为占位符: {samples[0][0]}")
            elif isinstance(evidence, EvidenceItem):
                sentence_id = evidence.sentence_id
                quote = evidence.quote
                
                # 如果sentence_id为空，尝试通过quote文本匹配
                if not sentence_id and quote:
                    quote_lower = quote.lower().strip()
                    for sid, text in sample_text_map.items():
                        if quote_lower in text.lower() or text.lower() in quote_lower:
                            sentence_id = sid
                            break
                
                if sentence_id and sentence_id in sample_ids:
                    validated_evidence.append(evidence.model_dump())
                elif sentence_id:
                    logger.warning(
                        f"证据条目中的sentence_id={sentence_id}不在输入样本中，已过滤"
                    )
                elif quote and samples:
                    # 使用第一个样本作为占位符
                    evidence_dict = evidence.model_dump()
                    evidence_dict["sentence_id"] = samples[0][0]
                    validated_evidence.append(evidence_dict)
                    logger.debug(f"无法匹配sentence_id，使用第一个样本作为占位符: {samples[0][0]}")
        
        # 如果所有证据都被过滤，至少保留第一个样本作为证据
        if not validated_evidence and samples:
            first_sample = samples[0]
            validated_evidence.append({
                "sentence_id": first_sample[0],
                "quote": first_sample[1][:200],  # 限制长度
                "why_representative": "作为代表样本"
            })
            logger.debug("所有证据都被过滤，使用第一个样本作为默认证据")
        
        insight["evidence_items"] = validated_evidence
        return insight
    
    def _generate_insight_simple(
        self,
        aspect_norm: str,
        cluster_id: str,
        cluster_size: int,
        neg_ratio: float,
        samples: List
    ) -> Dict[str, Any]:
        """生成简单洞察（fallback实现）"""
        priority = "high" if neg_ratio > 0.5 and cluster_size > 20 else "medium"
        if cluster_size < 10:
            priority = "low"
        
        cluster_name = f"{aspect_norm} issue cluster {cluster_id}"
        summary = f"Cluster contains {cluster_size} issues related to {aspect_norm}. "
        summary += f"Negative sentiment ratio: {neg_ratio:.2%}."
        
        evidence_items = []
        if samples:
            for sample in samples[:5]:
                evidence_items.append({
                    "sentence_id": sample[0],
                    "quote": sample[1][:200],  # 限制长度
                    "why_representative": "代表样本"
                })
        
        action_items = [
            {
                "action": f"Investigate {aspect_norm} issues",
                "urgency": priority,
                "owner_team": "product"
            }
        ]
        
        return {
            "cluster_name": cluster_name,
            "summary": summary,
            "priority": priority,
            "evidence_items": evidence_items,
            "action_items": action_items,
            "confidence": 0.5  # 简化版本置信度较低
        }

