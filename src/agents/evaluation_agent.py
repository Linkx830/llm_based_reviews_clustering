# 新增或修改我时需要修改这个文件夹中的README.md文件
"""EvaluationAgent - 自动指标 + 人工抽样包"""
from typing import Dict, Any
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationAgent(BaseAgent):
    """
    EvaluationAgent
    
    职责：
    - 计算自动评估指标
    - 生成人工评估抽样包
    
    输入表：各中间表
    输出表：evaluation_metrics
    """
    
    def process(self) -> Dict[str, Any]:
        """计算评估指标"""
        logger.info(f"开始执行EvaluationAgent，run_id={self.run_id}")
        
        # 判断是否使用传统模式
        use_traditional = "traditional" in self.pipeline_version.lower()
        if use_traditional:
            logger.info("检测到传统模式，使用传统表")
        
        metrics = {}
        
        # 覆盖率指标
        logger.info("计算覆盖率指标...")
        metrics.update(self._compute_coverage_metrics(use_traditional))
        logger.info(f"覆盖率指标: {metrics}")
        
        # 插入evaluation_metrics
        logger.info("插入评估指标到数据库...")
        table_manager = TableManager(self.db)
        for metric_name, metric_value in metrics.items():
            insert_query = f"""
                INSERT INTO {table_manager.EVALUATION_METRICS}
                (run_id, pipeline_version, data_slice_id, created_at,
                 metric_name, metric_value, metric_scope)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            self.db.execute_write(insert_query, {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id,
                "created_at": self.version_fields["created_at"],
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_scope": "overall"
            })
        
        logger.info(f"EvaluationAgent完成: 共计算 {len(metrics)} 个指标")
        
        return {
            "status": "success",
            "metrics": metrics,
            "table": table_manager.EVALUATION_METRICS
        }
    
    def _compute_coverage_metrics(self, use_traditional: bool = False) -> Dict[str, float]:
        """计算覆盖率指标"""
        import json
        
        # 根据模式选择表
        if use_traditional:
            raw_table = TableManager.ASPECT_SENTIMENT_RAW_TRADITIONAL
            valid_table = TableManager.ASPECT_SENTIMENT_VALID_TRADITIONAL
        else:
            raw_table = TableManager.ASPECT_SENTIMENT_RAW
            valid_table = TableManager.ASPECT_SENTIMENT_VALID
        
        # 总句子数
        logger.debug("查询总句子数...")
        total_sentences_query = f"""
            SELECT COUNT(*) FROM {TableManager.REVIEW_SENTENCES}
            WHERE run_id = ?
        """
        total_sentences = self.db.execute_read(
            total_sentences_query, {"run_id": self.run_id}
        )[0][0]
        logger.debug(f"总句子数: {total_sentences}")
        
        # 候选观点句数
        logger.debug("查询候选观点句数...")
        candidate_query = f"""
            SELECT COUNT(*) FROM {TableManager.OPINION_CANDIDATES}
            WHERE run_id = ? AND is_candidate = true
        """
        candidate_count = self.db.execute_read(
            candidate_query, {"run_id": self.run_id}
        )[0][0]
        logger.debug(f"候选观点句数: {candidate_count}")
        
        # 成功抽取的句子数（sentence_id去重）
        logger.debug("查询成功抽取的句子数...")
        if use_traditional:
            # 传统方法：所有记录都算成功（没有parse_status字段）
            success_sentences_query = f"""
                SELECT COUNT(DISTINCT sentence_id) FROM {raw_table}
                WHERE run_id = ?
            """
        else:
            # LLM方法：需要parse_status = 'SUCCESS'
            success_sentences_query = f"""
                SELECT COUNT(DISTINCT sentence_id) FROM {raw_table}
                WHERE run_id = ? AND parse_status = 'SUCCESS'
            """
        success_sentence_count = self.db.execute_read(
            success_sentences_query, {"run_id": self.run_id}
        )[0][0]
        logger.debug(f"成功抽取的句子数: {success_sentence_count}")
        
        # 成功抽取的总aspect数
        logger.debug("统计成功抽取的总aspect数...")
        if use_traditional:
            # 传统方法：直接统计记录数（每条记录一个aspect）
            aspects_query = f"""
                SELECT sentence_id FROM {raw_table}
                WHERE run_id = ?
            """
            aspect_records = self.db.execute_read(aspects_query, {"run_id": self.run_id})
            total_aspects_extracted = len(aspect_records)
            parsed_count = total_aspects_extracted
            failed_parse_count = 0
            sentences_with_aspects = len(set(row[0] for row in aspect_records))
            sentences_without_aspects = 0
        else:
            # LLM方法：从llm_output中统计
            aspects_query = f"""
                SELECT llm_output, sentence_id FROM {raw_table}
                WHERE run_id = ? AND parse_status = 'SUCCESS'
            """
            aspect_records = self.db.execute_read(aspects_query, {"run_id": self.run_id})
            total_aspects_extracted = 0
            parsed_count = 0
            failed_parse_count = 0
            sentences_with_aspects = 0
            sentences_without_aspects = 0
            
            # 使用set确保每个sentence_id只统计一次（防止重复）
            processed_sentence_ids = set()
            
            for (llm_output_json, sentence_id) in aspect_records:
                # 检查是否已经处理过这个sentence_id（防止重复统计）
                if sentence_id in processed_sentence_ids:
                    logger.warning(f"发现重复的sentence_id: {sentence_id}，跳过重复统计")
                    continue
                processed_sentence_ids.add(sentence_id)
                
                try:
                    if isinstance(llm_output_json, str):
                        output = json.loads(llm_output_json)
                    else:
                        output = llm_output_json
                    if isinstance(output, dict) and "aspects" in output:
                        aspects = output.get("aspects", [])
                        aspect_count = len(aspects) if isinstance(aspects, list) else 0
                        total_aspects_extracted += aspect_count
                        parsed_count += 1
                        if aspect_count == 0:
                            sentences_without_aspects += 1
                            logger.debug(f"句子 {sentence_id} 的aspect列表为空（has_opinion可能为false）")
                        else:
                            sentences_with_aspects += 1
                    else:
                        failed_parse_count += 1
                        logger.debug(f"句子 {sentence_id} 的llm_output格式异常: {type(output)}")
                except (json.JSONDecodeError, TypeError) as e:
                    failed_parse_count += 1
                    logger.debug(f"解析句子 {sentence_id} 的llm_output失败: {e}")
                    continue
        
        logger.info(
            f"成功抽取的总aspect数: {total_aspects_extracted} "
            f"(成功解析: {parsed_count}, 解析失败: {failed_parse_count}, "
            f"有aspect的句子: {sentences_with_aspects}, 无aspect的句子: {sentences_without_aspects})"
        )
        
        # VALID记录数（aspect级别）
        logger.debug("查询VALID记录数...")
        valid_query = f"""
            SELECT COUNT(*) FROM {valid_table}
            WHERE run_id = ? AND validity_label = 'VALID'
        """
        valid_count = self.db.execute_read(
            valid_query, {"run_id": self.run_id}
        )[0][0]
        logger.debug(f"VALID记录数: {valid_count}")
        
        # 诊断：检查是否有真正的重复（sentence_id+aspect+issue组合）
        true_duplicate_check_query = f"""
            SELECT sentence_id, aspect_norm, issue_norm, COUNT(*) as cnt
            FROM {valid_table}
            WHERE run_id = ? AND validity_label = 'VALID'
            GROUP BY sentence_id, aspect_norm, issue_norm
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT 10
        """
        true_duplicates = self.db.execute_read(true_duplicate_check_query, {"run_id": self.run_id})
        if true_duplicates:
            logger.warning(
                f"⚠️ 发现 {len(true_duplicates)} 个真正的重复记录（相同的sentence_id+aspect_norm+issue_norm组合）！"
                f"这可能是因为Judge阶段被多次执行，或者数据重复插入。"
            )
            for sentence_id, aspect_norm, issue_norm, cnt in true_duplicates[:5]:
                logger.warning(
                    f"  重复: sentence_id={sentence_id}, aspect_norm={aspect_norm}, "
                    f"issue_norm={issue_norm}, 出现次数={cnt}"
                )
        
        # 诊断：检查是否有重复的sentence_id+aspect组合（可能因为有不同的issue_norm，这是正常的）
        duplicate_check_query = f"""
            SELECT sentence_id, aspect_norm, COUNT(*) as cnt,
                   COUNT(DISTINCT issue_norm) as distinct_issues
            FROM {valid_table}
            WHERE run_id = ? AND validity_label = 'VALID'
            GROUP BY sentence_id, aspect_norm
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT 10
        """
        duplicates = self.db.execute_read(duplicate_check_query, {"run_id": self.run_id})
        if duplicates:
            # 区分两种情况：1) 有不同issue_norm（正常） 2) 有相同issue_norm（异常）
            normal_cases = []
            abnormal_cases = []
            for sentence_id, aspect_norm, cnt, distinct_issues in duplicates:
                if distinct_issues == cnt:
                    # 所有记录都有不同的issue_norm，这是正常的
                    normal_cases.append((sentence_id, aspect_norm, cnt, distinct_issues))
                else:
                    # 有相同的issue_norm，这是异常的
                    abnormal_cases.append((sentence_id, aspect_norm, cnt, distinct_issues))
            
            if abnormal_cases:
                logger.warning(
                    f"⚠️ 发现 {len(abnormal_cases)} 个异常的(sentence_id, aspect_norm)组合！"
                    f"这些组合中有相同的issue_norm，可能是真正的重复。"
                )
                for sentence_id, aspect_norm, cnt, distinct_issues in abnormal_cases[:5]:
                    logger.warning(
                        f"  异常: sentence_id={sentence_id}, aspect_norm={aspect_norm}, "
                        f"总记录数={cnt}, 不同issue数={distinct_issues}"
                    )
            
            if normal_cases and len(normal_cases) > len(abnormal_cases):
                logger.debug(
                    f"发现 {len(normal_cases)} 个(sentence_id, aspect_norm)组合有多个记录，"
                    f"但都有不同的issue_norm，这是正常的（一个句子可以针对同一个aspect提出多个问题）。"
                )
        
        # 诊断：检查是否有重复的sentence_id（同一个句子被处理多次）
        sentence_duplicate_query = f"""
            SELECT sentence_id, COUNT(*) as cnt
            FROM {valid_table}
            WHERE run_id = ? AND validity_label = 'VALID'
            GROUP BY sentence_id
            HAVING COUNT(*) > 10
            ORDER BY cnt DESC
            LIMIT 10
        """
        sentence_duplicates = self.db.execute_read(sentence_duplicate_query, {"run_id": self.run_id})
        if sentence_duplicates:
            logger.warning(
                f"⚠️ 发现 {len(sentence_duplicates)} 个sentence_id有超过10条VALID记录！"
                f"这可能表明数据重复或处理异常。"
            )
            for sentence_id, cnt in sentence_duplicates[:5]:
                logger.warning(f"  异常: sentence_id={sentence_id}, VALID记录数={cnt}")
        
        # 诊断：统计每个原始aspect对应的VALID记录数
        if total_aspects_extracted > 0:
            avg_valid_per_aspect = valid_count / total_aspects_extracted
            if avg_valid_per_aspect > 2.0:
                logger.warning(
                    f"⚠️ 平均每个原始aspect产生 {avg_valid_per_aspect:.2f} 个VALID记录，"
                    f"这异常高（正常应该≤1.0）。可能原因："
                    f"1. aspect_sentiment_raw表中有重复记录"
                    f"2. aspect_sentiment_valid表中有重复插入"
                    f"3. Judge阶段被多次执行"
                )
        
        # 有至少一个VALID aspect的句子数
        logger.debug("查询有VALID aspect的句子数...")
        valid_sentences_query = f"""
            SELECT COUNT(DISTINCT sentence_id) FROM {valid_table}
            WHERE run_id = ? AND validity_label = 'VALID'
        """
        valid_sentence_count = self.db.execute_read(
            valid_sentences_query, {"run_id": self.run_id}
        )[0][0]
        logger.debug(f"有VALID aspect的句子数: {valid_sentence_count}")
        
        # 计算有效aspect率，添加详细日志和异常检测
        if total_aspects_extracted > 0:
            valid_rate = valid_count / total_aspects_extracted
            if valid_rate > 1.0:
                logger.warning(
                    f"⚠️ 有效aspect率异常: {valid_rate:.2%} > 100%! "
                    f"VALID记录数({valid_count}) > 成功抽取的aspect数({total_aspects_extracted})。"
                    f"差异: {valid_count - total_aspects_extracted}。"
                    f"这可能是因为："
                    f"1. 某些aspect在Judge阶段被拆分（一个aspect变成多个VALID记录）"
                    f"2. 统计不完整（某些成功抽取的记录未被统计）"
                    f"3. 数据重复（某些记录被重复处理）"
                )
                # 如果超过100%，说明统计有问题，使用min(valid_rate, 1.0)作为上限
                # 但记录原始值用于诊断
                original_valid_rate = valid_rate
                valid_rate = min(valid_rate, 1.0)
                logger.warning(
                    f"已将有效aspect率从 {original_valid_rate:.2%} 截断为 {valid_rate:.2%} "
                    f"（上限100%）"
                )
        else:
            valid_rate = 0.0
            if valid_count > 0:
                logger.warning(
                    f"⚠️ 成功抽取的aspect数为0，但VALID记录数为{valid_count}，数据不一致！"
                    f"这可能表明数据表之间存在不一致，需要检查数据完整性。"
                )
        
        return {
            "candidate_coverage": candidate_count / total_sentences if total_sentences > 0 else 0.0,
            "extraction_success_rate": success_sentence_count / candidate_count if candidate_count > 0 else 0.0,
            "valid_rate": valid_rate,
            "valid_sentence_rate": valid_sentence_count / success_sentence_count if success_sentence_count > 0 else 0.0
        }

