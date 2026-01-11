# 新增或修改我时需要修改这个文件夹中的README.md文件
"""DataSelectorAgent - 数据切片与实验样本固定"""
from typing import Dict, Any, Optional, List
import hashlib
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..storage.version_fields import VersionFields
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataSelectorAgent(BaseAgent):
    """
    DataSelectorAgent
    
    职责：
    - 从reviews和meta中选择研究范围（类目/商品/时间）
    - 按parent_asin join元数据关键字段
    - 固定实验样本集，确保可复现
    
    输入表：reviews, meta
    输出表：selected_reviews
    """
    
    def process(
        self,
        main_category: Optional[str] = None,
        parent_asin: Optional[str] = None,
        time_window: Optional[tuple] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        选择数据切片
        """
        logger.info(
            f"开始执行DataSelectorAgent: category={main_category}, "
            f"asin={parent_asin}, limit={limit}"
        )
        
        """
        
        Args:
            main_category: 主类目
            parent_asin: 商品父ID
            time_window: 时间窗口 (start_timestamp, end_timestamp)
            filters: 过滤条件（如verified_purchase=True）
            limit: 限制数量
        
        Returns:
            处理结果统计
        """
        # 构建查询（使用?占位符，DuckDB需要位置参数）
        query_parts = [
            "SELECT",
            "r.parent_asin, r.asin, r.user_id, r.timestamp, r.rating,",
            "r.verified_purchase, r.helpful_vote, r.title as review_title,",
            "r.text as review_text, m.main_category, m.title as product_title",
            "FROM reviews r",
            "LEFT JOIN meta m ON r.parent_asin = m.parent_asin",
            "WHERE 1=1"
        ]
        params = []
        
        if main_category:
            query_parts.append("AND m.main_category = ?")
            params.append(main_category)
        
        if parent_asin:
            query_parts.append("AND r.parent_asin = ?")
            params.append(parent_asin)
        
        if time_window:
            query_parts.append("AND r.timestamp >= ? AND r.timestamp <= ?")
            params.append(time_window[0])
            params.append(time_window[1])
        
        if filters:
            if filters.get("verified_purchase") is not None:
                query_parts.append("AND r.verified_purchase = ?")
                params.append(filters["verified_purchase"])
            if filters.get("min_helpful_vote") is not None:
                query_parts.append("AND r.helpful_vote >= ?")
                params.append(filters["min_helpful_vote"])
        
        if limit:
            query_parts.append(f"LIMIT {limit}")
        
        query = " ".join(query_parts)
        
        # 执行查询（DuckDB使用位置参数列表）
        logger.info(f"执行数据查询，参数: {params}")
        results = self.db.execute_read(query, params if params else None)
        if len(results) == 0:
            logger.error(f"商品{parent_asin}无评论结果")
            raise ValueError("未查询到结果")
        logger.info(f"查询结果数量: {len(results)}")
        
        # 生成review_pk并插入selected_reviews表
        table_manager = TableManager(self.db)
        inserted_count = 0
        
        for row in results:
            # 解析查询结果字段（按SELECT顺序）
            # row[0]: parent_asin, row[1]: asin, row[2]: user_id, row[3]: timestamp,
            # row[4]: rating, row[5]: verified_purchase, row[6]: helpful_vote,
            # row[7]: review_title, row[8]: review_text, row[9]: main_category, row[10]: product_title
            parent_asin = row[0]
            asin = row[1]
            user_id = row[2]
            timestamp_raw = row[3]  # 可能是datetime对象或整数
            rating = row[4]
            verified_purchase = row[5]
            helpful_vote = row[6]
            review_title = row[7]
            review_text = row[8]
            main_category = row[9]
            product_title = row[10]
            
            # 转换timestamp为整数（Unix时间戳）
            # 如果已经是整数，直接使用；如果是datetime对象，转换为时间戳
            if timestamp_raw is None:
                timestamp = None
            elif isinstance(timestamp_raw, int):
                timestamp = timestamp_raw
            elif hasattr(timestamp_raw, 'timestamp'):  # datetime对象
                timestamp = int(timestamp_raw.timestamp())
            else:
                # 尝试直接转换
                timestamp = int(timestamp_raw)
            
            # 生成review_pk（确定性键）
            pk_parts = [
                str(parent_asin),
                str(user_id) if user_id else "",
                str(timestamp) if timestamp is not None else "",
                str(review_title) if review_title else "",
                str(review_text)[:100] if review_text else ""
            ]
            pk_content = "|".join(pk_parts)
            review_pk = hashlib.md5(pk_content.encode()).hexdigest()
            
            # 插入selected_reviews
            # 严格按照INSERT语句中的字段顺序排列参数
            insert_query = f"""
                INSERT INTO {table_manager.SELECTED_REVIEWS}
                (run_id, pipeline_version, data_slice_id, created_at,
                 review_pk, parent_asin, asin, user_id, timestamp, rating,
                 verified_purchase, helpful_vote, review_title, review_text,
                 main_category, product_title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            # 使用列表参数，严格按照INSERT字段顺序
            # DuckDB可以直接接受datetime对象（参考logging_tool.py的实现）
            params = [
                self.run_id,                                    # 1. run_id
                self.pipeline_version,                          # 2. pipeline_version
                self.data_slice_id,                            # 3. data_slice_id
                self.version_fields["created_at"],              # 4. created_at (TIMESTAMP)
                review_pk,                                     # 5. review_pk
                parent_asin,                                   # 6. parent_asin
                asin,                                          # 7. asin
                user_id,                                       # 8. user_id
                int(timestamp) if timestamp is not None else None,  # 9. timestamp (INTEGER)
                float(rating) if rating is not None else None,      # 10. rating
                bool(verified_purchase) if verified_purchase is not None else None,  # 11. verified_purchase
                int(helpful_vote) if helpful_vote is not None else None,  # 12. helpful_vote
                review_title,                                  # 13. review_title
                review_text,                                   # 14. review_text
                main_category,                                 # 15. main_category
                product_title                                  # 16. product_title
            ]
            self.db.execute_write(insert_query, params)
            inserted_count += 1
            
            # 每100条记录输出一次进度
            if inserted_count % 100 == 0:
                logger.info(f"已插入 {inserted_count} 条记录...")
        
        logger.info(f"DataSelectorAgent完成: 共插入 {inserted_count} 条记录")
        
        # 生成data_slice_id
        data_slice_id = VersionFields.generate_data_slice_id(
            main_category=main_category,
            parent_asin=parent_asin,
            time_window=time_window,
            filters=filters
        )
        
        return {
            "status": "success",
            "selected_count": inserted_count,
            "data_slice_id": data_slice_id,
            "table": table_manager.SELECTED_REVIEWS
        }

