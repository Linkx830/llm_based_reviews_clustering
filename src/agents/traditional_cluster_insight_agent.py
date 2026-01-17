# 新增或修改我时需要修改这个文件夹中的README.md文件
"""TraditionalClusterInsightAgent - 传统模板化洞察（无LLM）"""
import json
from typing import Dict, Any, List, Optional
from collections import Counter
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TraditionalClusterInsightAgent(BaseAgent):
    """
    TraditionalClusterInsightAgent
    
    职责：
    - 不使用LLM，对每个簇生成模板化洞察
    - 簇命名、摘要、证据、建议、优先级
    
    输入表：cluster_stats_traditional, aspect_sentiment_valid_traditional, review_sentences
    输出表：cluster_reports_traditional
    """
    
    def __init__(
        self,
        *args,
        method_version: str = "v1.0",
        evidence_count: int = 5,
        action_rule_table: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.method_version = method_version
        self.evidence_count = evidence_count
        
        # 初始化action规则表（关键词→建议模板）
        if action_rule_table is None:
            self.action_rule_table = self._init_default_action_rules()
        else:
            self.action_rule_table = action_rule_table
    
    def _init_default_action_rules(self) -> Dict[str, Dict[str, Any]]:
        """初始化默认action规则表"""
        return {
            "battery": {
                "action": "优化电池续航和充电速度",
                "owner_team": "product",
                "expected_impact": "提升用户满意度，减少退货率",
                "validation_plan": "A/B测试新电池方案，收集用户反馈"
            },
            "screen": {
                "action": "改进屏幕显示质量和耐用性",
                "owner_team": "design",
                "expected_impact": "提升产品竞争力",
                "validation_plan": "抽检屏幕质量，对比竞品"
            },
            "quality": {
                "action": "加强质量控制和质检流程",
                "owner_team": "quality",
                "expected_impact": "降低缺陷率，提升品牌形象",
                "validation_plan": "建立质量监控体系，定期抽检"
            },
            "price": {
                "action": "优化定价策略或提升性价比",
                "owner_team": "product",
                "expected_impact": "提升销量和市场份额",
                "validation_plan": "市场调研，分析竞品定价"
            },
            "default": {
                "action": "深入调研用户反馈，制定改进计划",
                "owner_team": "product",
                "expected_impact": "提升用户满意度",
                "validation_plan": "用户访谈，数据分析"
            }
        }
    
    def process(self) -> Dict[str, Any]:
        """生成簇洞察"""
        logger.info(f"开始执行TraditionalClusterInsightAgent，run_id={self.run_id}")
        
        # 读取cluster_stats_traditional
        query = f"""
            SELECT aspect_norm, cluster_id, cluster_size, neg_ratio,
                   representative_sentence_ids
            FROM {TableManager.CLUSTER_STATS_TRADITIONAL}
            WHERE run_id = ?
            ORDER BY cluster_size DESC
            LIMIT 50
        """
        clusters = self.db.execute_read(query, {"run_id": self.run_id})
        
        logger.info(f"读取到 {len(clusters)} 个簇，开始生成洞察")
        
        table_manager = TableManager(self.db)
        processed_count = 0
        
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
                    SELECT rs.sentence_id, rs.target_sentence, asv.issue_norm, asv.sentiment
                    FROM {TableManager.REVIEW_SENTENCES} rs
                    JOIN {TableManager.ASPECT_SENTIMENT_VALID_TRADITIONAL} asv
                        ON rs.sentence_id = asv.sentence_id AND rs.run_id = asv.run_id
                    WHERE rs.sentence_id IN ({placeholders}) AND rs.run_id = ?
                    LIMIT {self.evidence_count}
                """
                params = list(rep_sentence_ids) + [self.run_id]
                samples = self.db.execute_read(sample_query, params)
            
            # 生成模板化洞察
            insight = self._generate_insight_template(
                aspect_norm, cluster_id, cluster_size, neg_ratio, samples
            )
            
            # 插入cluster_reports_traditional
            insert_query = f"""
                INSERT INTO {table_manager.CLUSTER_REPORTS_TRADITIONAL}
                (run_id, pipeline_version, data_slice_id, created_at,
                 method_version, aspect_norm, cluster_id, cluster_name, summary,
                 priority, priority_rationale, evidence_items, action_items, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # 转换evidence_items和action_items为JSON
            evidence_json = json.dumps(
                insight.get("evidence_items", []),
                ensure_ascii=False
            )
            action_json = json.dumps(
                insight.get("action_items", []),
                ensure_ascii=False
            )
            
            self.db.execute_write(insert_query, {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id,
                "created_at": self.version_fields["created_at"],
                "method_version": self.method_version,
                "aspect_norm": aspect_norm,
                "cluster_id": str(cluster_id),
                "cluster_name": insight.get("cluster_name", ""),
                "summary": insight.get("summary", ""),
                "priority": insight.get("priority", "medium"),
                "priority_rationale": insight.get("priority_rationale"),
                "evidence_items": evidence_json,
                "action_items": action_json,
                "confidence": insight.get("confidence", 0.7)
            })
            processed_count += 1
        
        logger.info(f"TraditionalClusterInsightAgent完成: processed={processed_count}")
        
        return {
            "status": "success",
            "processed_count": processed_count,
            "table": table_manager.CLUSTER_REPORTS_TRADITIONAL
        }
    
    def _generate_insight_template(
        self,
        aspect_norm: str,
        cluster_id: str,
        cluster_size: int,
        neg_ratio: float,
        samples: List
    ) -> Dict[str, Any]:
        """生成模板化洞察"""
        
        # 1. 簇命名：{aspect_norm}: {top_issue_phrase}
        cluster_name = self._generate_cluster_name(aspect_norm, samples)
        
        # 2. 摘要：模板化描述
        summary = self._generate_summary(aspect_norm, cluster_size, neg_ratio, samples)
        
        # 3. 优先级：基于规则计算
        priority, priority_rationale = self._calculate_priority(cluster_size, neg_ratio)
        
        # 4. 证据条目：从representative samples提取
        evidence_items = self._generate_evidence_items(samples)
        
        # 5. 建议条目：基于规则映射
        action_items = self._generate_action_items(aspect_norm, cluster_size, neg_ratio)
        
        # 6. 置信度：基于簇内一致性和分离度估算
        confidence = self._estimate_confidence(cluster_size, neg_ratio)
        
        return {
            "cluster_name": cluster_name,
            "summary": summary,
            "priority": priority,
            "priority_rationale": priority_rationale,
            "evidence_items": evidence_items,
            "action_items": action_items,
            "confidence": confidence
        }
    
    def _generate_cluster_name(self, aspect_norm: str, samples: List) -> str:
        """生成簇名称"""
        if not samples:
            return f"{aspect_norm} issue cluster"
        
        # 提取最常见的issue词
        issue_words = []
        for sample in samples:
            if len(sample) > 2:
                issue_norm = sample[2] or ""
                issue_words.extend(issue_norm.lower().split()[:3])
        
        if issue_words:
            issue_counter = Counter(issue_words)
            top_issue = issue_counter.most_common(1)[0][0]
            return f"{aspect_norm}: {top_issue}"
        
        return f"{aspect_norm} issue cluster"
    
    def _generate_summary(
        self,
        aspect_norm: str,
        cluster_size: int,
        neg_ratio: float,
        samples: List
    ) -> str:
        """生成摘要"""
        # 提取top terms
        top_terms = []
        for sample in samples[:5]:
            if len(sample) > 2:
                issue_norm = sample[2] or ""
                if issue_norm:
                    top_terms.append(issue_norm)
        
        top_terms_str = ", ".join(top_terms[:3]) if top_terms else "various issues"
        
        summary = (
            f"用户在 {aspect_norm} 方面主要反馈 {top_terms_str}，"
            f"负面率 {neg_ratio:.1%}，样本量 {cluster_size}。"
        )
        
        if neg_ratio > 0.7:
            summary += "负面反馈占比较高，需要重点关注。"
        elif neg_ratio < 0.3:
            summary += "整体反馈较为正面。"
        
        return summary
    
    def _calculate_priority(
        self,
        cluster_size: int,
        neg_ratio: float
    ) -> tuple[str, str]:
        """计算优先级"""
        # 优先级规则：count × neg_ratio × trend_weight
        priority_score = cluster_size * neg_ratio
        
        if priority_score > 20 or (cluster_size > 50 and neg_ratio > 0.7):
            priority = "high"
            rationale = f"规模大({cluster_size})且负面率高({neg_ratio:.1%})"
        elif priority_score > 10 or (cluster_size > 20 and neg_ratio > 0.5):
            priority = "medium"
            rationale = f"规模中等({cluster_size})，负面率{neg_ratio:.1%}"
        else:
            priority = "low"
            rationale = f"规模较小({cluster_size})或负面率较低({neg_ratio:.1%})"
        
        return priority, rationale
    
    def _generate_evidence_items(self, samples: List) -> List[Dict[str, Any]]:
        """生成证据条目"""
        evidence_items = []
        
        for sample in samples[:self.evidence_count]:
            if len(sample) >= 2:
                sentence_id = sample[0]
                target_sentence = sample[1]
                evidence_items.append({
                    "sentence_id": sentence_id,
                    "quote": target_sentence[:200],  # 限制长度
                    "why_representative": "代表样本"
                })
        
        return evidence_items
    
    def _generate_action_items(
        self,
        aspect_norm: str,
        cluster_size: int,
        neg_ratio: float
    ) -> List[Dict[str, Any]]:
        """生成建议条目"""
        action_items = []
        
        # 查找匹配的action规则
        aspect_lower = aspect_norm.lower()
        matched_rule = None
        
        for key, rule in self.action_rule_table.items():
            if key in aspect_lower or aspect_lower in key:
                matched_rule = rule
                break
        
        if not matched_rule:
            matched_rule = self.action_rule_table.get("default", {})
        
        # 构建action item
        urgency = "high" if neg_ratio > 0.7 else "medium" if neg_ratio > 0.5 else "low"
        
        action_items.append({
            "action": matched_rule.get("action", "深入调研用户反馈，制定改进计划"),
            "owner_team": matched_rule.get("owner_team", "product"),
            "expected_impact": matched_rule.get("expected_impact", "提升用户满意度"),
            "validation_plan": matched_rule.get("validation_plan", "用户访谈，数据分析"),
            "urgency": urgency
        })
        
        # 如果负面率高，添加额外的action
        if neg_ratio > 0.7:
            action_items.append({
                "action": "优先处理负面反馈，制定快速响应方案",
                "owner_team": "customer_service",
                "expected_impact": "降低用户流失率",
                "validation_plan": "跟踪用户反馈变化趋势",
                "urgency": "high"
            })
        
        return action_items
    
    def _estimate_confidence(self, cluster_size: int, neg_ratio: float) -> float:
        """估算置信度"""
        # 基于簇大小和负面率一致性估算置信度
        base_confidence = 0.7
        
        # 簇越大，置信度越高
        if cluster_size > 20:
            base_confidence += 0.1
        elif cluster_size > 10:
            base_confidence += 0.05
        
        # 负面率越极端（接近0或1），置信度越高（说明一致性高）
        if neg_ratio > 0.8 or neg_ratio < 0.2:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)



