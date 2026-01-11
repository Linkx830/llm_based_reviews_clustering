# 新增或修改我时需要修改这个文件夹中的README.md文件
"""ReportAssemblerAgent - 报告组装"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReportAssemblerAgent(BaseAgent):
    """
    ReportAssemblerAgent
    
    职责：
    - 输出专业Markdown报告
    - 包含聚类效果参数和评估指标
    
    输入表：cluster_reports, cluster_stats, evaluation_metrics等
    输出：final_report.md
    """
    
    def process(self, output_dir: Path) -> Dict[str, Any]:
        """生成报告"""
        logger.info(
            f"开始执行ReportAssemblerAgent，run_id={self.run_id}, "
            f"output_dir={output_dir}"
        )
        
        # 读取簇报告和统计信息（JOIN cluster_stats获取聚类效果参数）
        clusters_query = f"""
            SELECT 
                cr.aspect_norm,
                cr.cluster_id,
                cr.cluster_name, 
                cr.summary, 
                cr.priority, 
                cr.evidence_items, 
                cr.action_items,
                cr.confidence,
                cs.cluster_size,
                cs.neg_ratio,
                cs.intra_cluster_distance,
                cs.inter_cluster_distance,
                cs.separation_ratio,
                cs.cohesion,
                cs.cluster_confidence,
                cs.sentiment_consistency,
                cs.top_terms,
                cs.representative_sentence_ids
            FROM {TableManager.CLUSTER_REPORTS} cr
            LEFT JOIN {TableManager.CLUSTER_STATS} cs
                ON cr.run_id = cs.run_id 
                AND cr.aspect_norm = cs.aspect_norm 
                AND cr.cluster_id = cs.cluster_id
            WHERE cr.run_id = ?
            ORDER BY 
                CASE cr.priority 
                    WHEN 'high' THEN 1 
                    WHEN 'medium' THEN 2 
                    WHEN 'low' THEN 3 
                    ELSE 4 
                END,
                COALESCE(cs.cluster_size, 0) DESC,
                cr.cluster_id
            LIMIT 20
        """
        clusters = self.db.execute_read(clusters_query, {"run_id": self.run_id})
        logger.info(f"读取到 {len(clusters)} 个簇报告")
        
        # 调试：检查第一个簇的数据结构
        if clusters:
            logger.debug(f"第一个簇的数据示例: {clusters[0]}")
            logger.debug(f"第一个簇的aspect_norm: {clusters[0][0] if len(clusters[0]) > 0 else 'N/A'}")
        
        # 读取评估指标
        metrics_query = f"""
            SELECT metric_name, metric_value, metric_scope, notes
            FROM {TableManager.EVALUATION_METRICS}
            WHERE run_id = ?
            ORDER BY metric_name
        """
        metrics = self.db.execute_read(metrics_query, {"run_id": self.run_id})
        logger.info(f"读取到 {len(metrics)} 个评估指标")
        
        # 读取基础统计信息
        stats = self._get_basic_stats()
        
        # 生成Markdown报告
        logger.info("生成Markdown报告内容...")
        report_content = self._generate_markdown(clusters, metrics, stats)
        
        # 写入文件
        report_path = output_dir / "final_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_content, encoding="utf-8")
        logger.info(f"报告已写入: {report_path}")
        
        return {
            "status": "success",
            "report_path": str(report_path),
            "cluster_count": len(clusters),
            "metric_count": len(metrics)
        }
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        """获取基础统计信息"""
        stats = {}
        
        # 总评论数
        try:
            query = f"SELECT COUNT(*) FROM {TableManager.SELECTED_REVIEWS} WHERE run_id = ?"
            result = self.db.execute_read(query, {"run_id": self.run_id})
            stats["total_reviews"] = result[0][0] if result else 0
        except Exception as e:
            logger.warning(f"获取总评论数失败: {e}")
            stats["total_reviews"] = 0
        
        # 总句子数
        try:
            query = f"SELECT COUNT(*) FROM {TableManager.REVIEW_SENTENCES} WHERE run_id = ?"
            result = self.db.execute_read(query, {"run_id": self.run_id})
            stats["total_sentences"] = result[0][0] if result else 0
        except Exception as e:
            logger.warning(f"获取总句子数失败: {e}")
            stats["total_sentences"] = 0
        
        # 候选观点句数
        try:
            query = f"""
                SELECT COUNT(*) FROM {TableManager.OPINION_CANDIDATES} 
                WHERE run_id = ? AND is_candidate = true
            """
            result = self.db.execute_read(query, {"run_id": self.run_id})
            stats["candidate_sentences"] = result[0][0] if result else 0
        except Exception as e:
            logger.warning(f"获取候选观点句数失败: {e}")
            stats["candidate_sentences"] = 0
        
        # VALID记录数
        try:
            query = f"""
                SELECT COUNT(*) FROM {TableManager.ASPECT_SENTIMENT_VALID} 
                WHERE run_id = ? AND validity_label = 'VALID'
            """
            result = self.db.execute_read(query, {"run_id": self.run_id})
            stats["valid_records"] = result[0][0] if result else 0
        except Exception as e:
            logger.warning(f"获取VALID记录数失败: {e}")
            stats["valid_records"] = 0
        
        # 簇总数
        try:
            query = f"""
                SELECT COUNT(DISTINCT cluster_id) FROM {TableManager.CLUSTER_STATS} 
                WHERE run_id = ? AND cluster_id != 'noise'
            """
            result = self.db.execute_read(query, {"run_id": self.run_id})
            stats["total_clusters"] = result[0][0] if result else 0
        except Exception as e:
            logger.warning(f"获取簇总数失败: {e}")
            stats["total_clusters"] = 0
        
        # 噪声点数量
        try:
            query = f"""
                SELECT COUNT(*) FROM {TableManager.ISSUE_CLUSTERS} 
                WHERE run_id = ? AND is_noise = true
            """
            result = self.db.execute_read(query, {"run_id": self.run_id})
            stats["noise_count"] = result[0][0] if result else 0
        except Exception as e:
            logger.warning(f"获取噪声点数量失败: {e}")
            stats["noise_count"] = 0
        
        # 聚类质量指标（平均值）
        try:
            query = f"""
                SELECT 
                    AVG(cluster_confidence) as avg_cluster_confidence,
                    AVG(separation_ratio) as avg_separation_ratio,
                    AVG(cohesion) as avg_cohesion,
                    AVG(sentiment_consistency) as avg_sentiment_consistency
                FROM {TableManager.CLUSTER_STATS}
                WHERE run_id = ? 
                AND cluster_confidence IS NOT NULL
            """
            result = self.db.execute_read(query, {"run_id": self.run_id})
            if result and result[0]:
                stats["avg_cluster_confidence"] = result[0][0] if result[0][0] is not None else None
                stats["avg_separation_ratio"] = result[0][1] if result[0][1] is not None else None
                stats["avg_cohesion"] = result[0][2] if result[0][2] is not None else None
                stats["avg_sentiment_consistency"] = result[0][3] if result[0][3] is not None else None
        except Exception as e:
            logger.warning(f"获取聚类质量指标失败: {e}")
        
        return stats
    
    def _generate_markdown(
        self, 
        clusters: List[tuple], 
        metrics: List[tuple],
        stats: Dict[str, Any]
    ) -> str:
        """生成Markdown内容"""
        lines = [
            "# 电商评论方面级情感分析报告",
            "",
            "## 1. 数据与前提条件",
            "",
            f"- **Run ID**: {self.run_id}",
            f"- **Pipeline Version**: {self.pipeline_version}",
            f"- **Data Slice ID**: {self.data_slice_id}",
            "",
            "### 数据统计",
            "",
            f"- 总评论数: {stats.get('total_reviews', 0):,}",
            f"- 总句子数: {stats.get('total_sentences', 0):,}",
            f"- 候选观点句数: {stats.get('candidate_sentences', 0):,}",
            f"- 有效抽取记录数: {stats.get('valid_records', 0):,}",
            f"- 簇总数: {stats.get('total_clusters', 0)}",
            f"- 噪声点数量: {stats.get('noise_count', 0):,}",
            "",
            "---",
            "",
            "## 2. 实验与评估",
            "",
            "### 2.1 覆盖率指标",
            "",
            "| 指标名称 | 数值 | 说明 |",
            "|---------|------|------|"
        ]
        
        # 添加评估指标
        if not metrics:
            lines.append("| *暂无评估指标* | - | 请检查EvaluationAgent是否已执行 |")
        else:
            # 定义指标的中文名称和说明
            metric_descriptions = {
                "candidate_coverage": ("候选观点句覆盖率", "候选观点句数 / 总句子数，表示被识别为观点候选的句子比例"),
                "extraction_success_rate": ("抽取成功率", "成功抽取的句子数 / 候选观点句数，表示LLM成功解析的句子比例"),
                "valid_rate": ("有效aspect率", "VALID aspect记录数 / 成功抽取的总aspect数，表示通过校验的aspect比例"),
                "valid_sentence_rate": ("有效句子率", "有至少一个VALID aspect的句子数 / 成功抽取的句子数，表示至少有一个有效aspect的句子比例"),
                "silhouette_score": ("轮廓系数", "聚类质量指标，范围[-1, 1]，越大越好"),
            }
            
            for metric_name, metric_value, metric_scope, notes in metrics:
                scope_str = f" ({metric_scope})" if metric_scope else ""
                # 使用预定义的说明，如果没有则使用notes
                if metric_name in metric_descriptions:
                    display_name, description = metric_descriptions[metric_name]
                    notes_str = description
                else:
                    display_name = metric_name
                    notes_str = notes if notes else ""
                
                # 格式化数值（如果是百分比类型的指标）
                if "coverage" in metric_name or "rate" in metric_name:
                    value_str = f"{metric_value:.2%}"
                else:
                    value_str = f"{metric_value:.4f}"
                
                lines.append(f"| {display_name}{scope_str} | {value_str} | {notes_str} |")
        
        lines.extend([
            "",
            "### 2.2 聚类效果参数",
            "",
            "| 参数名称 | 数值 | 说明 |",
            "|---------|------|------|",
            f"| 簇总数 | {stats.get('total_clusters', 0)} | 有效簇的数量（不含噪声点） |",
            f"| 噪声点数量 | {stats.get('noise_count', 0):,} | 聚类算法识别为噪声点的记录数 |",
        ])
        
        # 计算噪声点比例
        total_clustered = stats.get('valid_records', 0)
        noise_count = stats.get('noise_count', 0)
        if total_clustered > 0:
            noise_ratio = noise_count / total_clustered
            lines.append(f"| 噪声点比例 | {noise_ratio:.2%} | 噪声点占总记录数的比例 |")
        
        # 添加聚类质量指标
        avg_cluster_confidence = stats.get('avg_cluster_confidence')
        avg_separation_ratio = stats.get('avg_separation_ratio')
        avg_cohesion = stats.get('avg_cohesion')
        avg_sentiment_consistency = stats.get('avg_sentiment_consistency')
        
        if avg_cluster_confidence is not None:
            lines.append(f"| 平均聚类置信度 | {avg_cluster_confidence:.4f} | 所有簇的平均置信度（0-1，越大越好） |")
        if avg_separation_ratio is not None:
            lines.append(f"| 平均分离度比率 | {avg_separation_ratio:.4f} | 簇间距离/簇内距离的平均值（越大越好） |")
        if avg_cohesion is not None:
            lines.append(f"| 平均紧密度 | {avg_cohesion:.4f} | 所有簇的平均紧密度（越大越好） |")
        if avg_sentiment_consistency is not None:
            lines.append(f"| 平均情感一致性 | {avg_sentiment_consistency:.2%} | 所有簇的平均情感一致性（越大越好） |")
        
        lines.extend([
            "",
            "---",
            "",
            "## 3. Top 痛点簇",
            "",
            "### 3.1 簇概览表",
            "",
            "| 排名 | 簇名称 | 方面 | 簇大小 | 负面率 | 优先级 | LLM置信度 | 聚类置信度 |",
            "|------|--------|------|--------|--------|--------|------------|------------|"
        ])
        
        # 添加簇概览
        for idx, cluster in enumerate(clusters, 1):
            try:
                (
                    aspect_norm, cluster_id, cluster_name, summary, priority,
                    evidence_items, action_items, llm_confidence,
                    cluster_size, neg_ratio, intra_cluster_distance, inter_cluster_distance,
                    separation_ratio, cohesion, cluster_confidence, sentiment_consistency,
                    top_terms, representative_sentence_ids
                ) = cluster
            except (ValueError, IndexError) as e:
                logger.error(f"解包簇数据失败 (索引 {idx}): {e}, 数据: {cluster}")
                # 尝试安全解包
                aspect_norm = cluster[0] if len(cluster) > 0 else "未知"
                cluster_id = cluster[1] if len(cluster) > 1 else "未知"
                cluster_name = cluster[2] if len(cluster) > 2 else "未知"
                summary = cluster[3] if len(cluster) > 3 else ""
                priority = cluster[4] if len(cluster) > 4 else "medium"
                evidence_items = cluster[5] if len(cluster) > 5 else None
                action_items = cluster[6] if len(cluster) > 6 else None
                llm_confidence = cluster[7] if len(cluster) > 7 else None
                cluster_size = cluster[8] if len(cluster) > 8 else None
                neg_ratio = cluster[9] if len(cluster) > 9 else None
                intra_cluster_distance = cluster[10] if len(cluster) > 10 else None
                inter_cluster_distance = cluster[11] if len(cluster) > 11 else None
                separation_ratio = cluster[12] if len(cluster) > 12 else None
                cohesion = cluster[13] if len(cluster) > 13 else None
                cluster_confidence = cluster[14] if len(cluster) > 14 else None
                sentiment_consistency = cluster[15] if len(cluster) > 15 else None
                top_terms = cluster[16] if len(cluster) > 16 else None
                representative_sentence_ids = cluster[17] if len(cluster) > 17 else None
            
            # 处理空值
            aspect_norm = aspect_norm if aspect_norm else "未知方面"
            cluster_name = cluster_name if cluster_name else "未命名簇"
            cluster_size_str = f"{cluster_size:,}" if cluster_size else "N/A"
            neg_ratio_str = f"{neg_ratio:.2%}" if neg_ratio is not None else "N/A"
            llm_confidence_str = f"{llm_confidence:.2f}" if llm_confidence is not None else "N/A"
            cluster_confidence_str = f"{cluster_confidence:.2f}" if cluster_confidence is not None else "N/A"
            
            lines.append(
                f"| {idx} | {cluster_name} | {aspect_norm} | {cluster_size_str} | "
                f"{neg_ratio_str} | {priority} | {llm_confidence_str} | {cluster_confidence_str} |"
            )
        
        lines.extend([
            "",
            "### 3.2 簇详细分析",
            ""
        ])
        
        # 添加每个簇的详细信息
        for idx, cluster in enumerate(clusters, 1):
            try:
                (
                    aspect_norm, cluster_id, cluster_name, summary, priority,
                    evidence_items, action_items, llm_confidence,
                    cluster_size, neg_ratio, intra_cluster_distance, inter_cluster_distance,
                    separation_ratio, cohesion, cluster_confidence, sentiment_consistency,
                    top_terms, representative_sentence_ids
                ) = cluster
            except (ValueError, IndexError) as e:
                logger.error(f"解包簇数据失败 (索引 {idx}): {e}, 数据: {cluster}")
                # 尝试安全解包
                aspect_norm = cluster[0] if len(cluster) > 0 else "未知"
                cluster_id = cluster[1] if len(cluster) > 1 else "未知"
                cluster_name = cluster[2] if len(cluster) > 2 else "未知"
                summary = cluster[3] if len(cluster) > 3 else ""
                priority = cluster[4] if len(cluster) > 4 else "medium"
                evidence_items = cluster[5] if len(cluster) > 5 else None
                action_items = cluster[6] if len(cluster) > 6 else None
                llm_confidence = cluster[7] if len(cluster) > 7 else None
                cluster_size = cluster[8] if len(cluster) > 8 else None
                neg_ratio = cluster[9] if len(cluster) > 9 else None
                intra_cluster_distance = cluster[10] if len(cluster) > 10 else None
                inter_cluster_distance = cluster[11] if len(cluster) > 11 else None
                separation_ratio = cluster[12] if len(cluster) > 12 else None
                cohesion = cluster[13] if len(cluster) > 13 else None
                cluster_confidence = cluster[14] if len(cluster) > 14 else None
                sentiment_consistency = cluster[15] if len(cluster) > 15 else None
                top_terms = cluster[16] if len(cluster) > 16 else None
                representative_sentence_ids = cluster[17] if len(cluster) > 17 else None
            
            # 处理空值
            aspect_norm = aspect_norm if aspect_norm else "未知方面"
            cluster_name = cluster_name if cluster_name else "未命名簇"
            cluster_id = cluster_id if cluster_id else "未知"
            summary = summary if summary else "无摘要"
            priority = priority if priority else "medium"
            
            lines.extend([
                f"#### {idx}. {cluster_name}",
                "",
                f"**方面**: {aspect_norm}",
                f"**簇ID**: {cluster_id}",
                f"**优先级**: {priority}",
                ""
            ])
            
            # 聚类效果参数
            lines.append("**聚类效果参数**:")
            lines.append("")
            if cluster_size is not None:
                lines.append(f"- 簇大小: {cluster_size:,} 条记录")
            if neg_ratio is not None:
                lines.append(f"- 负面率: {neg_ratio:.2%}")
            if llm_confidence is not None:
                lines.append(f"- LLM置信度: {llm_confidence:.2f}")
            if cluster_confidence is not None:
                lines.append(f"- 聚类置信度: {cluster_confidence:.2f} (基于簇内一致性和分离度)")
            if intra_cluster_distance is not None:
                lines.append(f"- 簇内平均距离: {intra_cluster_distance:.4f}")
            if inter_cluster_distance is not None:
                lines.append(f"- 簇间最小距离: {inter_cluster_distance:.4f}")
            if separation_ratio is not None:
                lines.append(f"- 分离度比率: {separation_ratio:.4f} (簇间距离/簇内距离，越大越好)")
            if cohesion is not None:
                lines.append(f"- 紧密度: {cohesion:.4f} (簇内距离的倒数，越大越好)")
            if sentiment_consistency is not None:
                lines.append(f"- 情感一致性: {sentiment_consistency:.2%} (簇内情感分布的均匀程度)")
            if top_terms:
                try:
                    terms = json.loads(top_terms) if isinstance(top_terms, str) else top_terms
                    if terms:
                        terms_str = ", ".join(terms[:10])  # 只显示前10个
                        lines.append(f"- 关键词: {terms_str}")
                except:
                    pass
            
            lines.extend([
                "",
                f"**摘要**: {summary}",
                "",
                "**证据**:",
                ""
            ])
            
            # 解析并显示证据
            try:
                if evidence_items:
                    evidence_list = json.loads(evidence_items) if isinstance(evidence_items, str) else evidence_items
                    if isinstance(evidence_list, list) and evidence_list:
                        for evidence in evidence_list[:5]:  # 只显示前5条
                            if isinstance(evidence, dict):
                                quote = evidence.get('quote', '')
                                sentence_id = evidence.get('sentence_id', '')
                                why = evidence.get('why_representative', '')
                                if quote:
                                    lines.append(f"- {quote}")
                                    if why:
                                        lines.append(f"  - *{why}*")
                            elif isinstance(evidence, str):
                                lines.append(f"- {evidence}")
                    else:
                        lines.append("- 详见数据库记录")
                else:
                    lines.append("- 详见数据库记录")
            except Exception as e:
                logger.warning(f"解析证据失败: {e}")
                lines.append("- 详见数据库记录")
            
            lines.extend([
                "",
                "**建议**:",
                ""
            ])
            
            # 解析并显示建议
            try:
                if action_items:
                    action_list = json.loads(action_items) if isinstance(action_items, str) else action_items
                    if isinstance(action_list, list) and action_list:
                        for action in action_list:
                            if isinstance(action, dict):
                                action_text = action.get('action', '')
                                owner = action.get('owner_team', '')
                                impact = action.get('expected_impact', '')
                                if action_text:
                                    lines.append(f"- {action_text}")
                                    if owner:
                                        lines.append(f"  - 责任方: {owner}")
                                    if impact:
                                        lines.append(f"  - 预期影响: {impact}")
                            elif isinstance(action, str):
                                lines.append(f"- {action}")
                    else:
                        lines.append("- 详见数据库记录")
                else:
                    lines.append("- 详见数据库记录")
            except Exception as e:
                logger.warning(f"解析建议失败: {e}")
                lines.append("- 详见数据库记录")
            
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "## 4. 失败分析与改进方向",
            "",
            "### 4.1 数据质量分析",
            ""
        ])
        
        # 计算一些质量指标
        total_sentences = stats.get('total_sentences', 0)
        candidate_sentences = stats.get('candidate_sentences', 0)
        valid_records = stats.get('valid_records', 0)
        
        if total_sentences > 0:
            candidate_rate = candidate_sentences / total_sentences
            lines.append(f"- 候选观点句覆盖率: {candidate_rate:.2%}")
        
        if candidate_sentences > 0:
            valid_rate = valid_records / candidate_sentences
            lines.append(f"- 有效抽取率: {valid_rate:.2%}")
        
        noise_count = stats.get('noise_count', 0)
        if valid_records > 0:
            noise_ratio = noise_count / valid_records
            lines.append(f"- 噪声点比例: {noise_ratio:.2%}")
        
        lines.extend([
            "",
            "### 4.2 改进建议",
            "",
            "- 优化观点候选过滤规则，提高候选句质量",
            "- 改进LLM抽取提示词，减少解析失败率",
            "- 优化聚类参数，减少噪声点比例",
            "- 增强方面归一化，提高簇内一致性",
            ""
        ])
        
        return "\n".join(lines)

