# 新增或修改我时需要修改这个文件夹中的README.md文件
"""DuckDB表管理模块"""
from typing import Dict, List, Optional
from .connection import DuckDBConnection


class TableManager:
    """
    DuckDB表管理
    
    职责：
    - 表存在性检查
    - 表结构定义与创建
    - run_id过滤视图
    - 表契约校验
    """
    
    # 表名常量
    SELECTED_REVIEWS = "selected_reviews"
    META_CONTEXT = "meta_context"
    NORMALIZED_REVIEWS = "normalized_reviews"
    REVIEW_SENTENCES = "review_sentences"
    OPINION_CANDIDATES = "opinion_candidates"
    ASPECT_SENTIMENT_RAW = "aspect_sentiment_raw"
    ASPECT_SENTIMENT_VALID = "aspect_sentiment_valid"
    EXTRACTION_ISSUES = "extraction_issues"
    ISSUE_CLUSTERS = "issue_clusters"
    CLUSTER_STATS = "cluster_stats"
    CLUSTER_REPORTS = "cluster_reports"
    EVALUATION_METRICS = "evaluation_metrics"
    RUN_LOG = "run_log"
    
    # 传统方法专用表
    ASPECT_SENTIMENT_RAW_TRADITIONAL = "aspect_sentiment_raw_traditional"
    ASPECT_SENTIMENT_VALID_TRADITIONAL = "aspect_sentiment_valid_traditional"
    EXTRACTION_ISSUES_TRADITIONAL = "extraction_issues_traditional"
    ISSUE_CLUSTERS_TRADITIONAL = "issue_clusters_traditional"
    CLUSTER_STATS_TRADITIONAL = "cluster_stats_traditional"
    CLUSTER_REPORTS_TRADITIONAL = "cluster_reports_traditional"
    
    def __init__(self, db_conn: DuckDBConnection):
        self.db = db_conn
    
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        query = """
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = ?
        """
        result = self.db.execute_read(query, {"table_name": table_name})
        return result[0][0] > 0 if result else False
    
    def create_all_tables(self):
        """创建所有中间表"""
        self.create_selected_reviews_table()
        self.create_meta_context_table()
        self.create_normalized_reviews_table()
        self.create_review_sentences_table()
        self.create_opinion_candidates_table()
        self.create_aspect_sentiment_raw_table()
        self.create_aspect_sentiment_valid_table()
        self.create_extraction_issues_table()
        self.create_issue_clusters_table()
        self.create_cluster_stats_table()
        self.create_cluster_reports_table()
        self.create_evaluation_metrics_table()
        self.create_run_log_table()
        # 传统方法专用表
        self.create_aspect_sentiment_raw_traditional_table()
        self.create_aspect_sentiment_valid_traditional_table()
        self.create_extraction_issues_traditional_table()
        self.create_issue_clusters_traditional_table()
        self.create_cluster_stats_traditional_table()
        self.create_cluster_reports_traditional_table()
    
    def create_selected_reviews_table(self):
        """创建selected_reviews表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            source_snapshot_at TIMESTAMP,
            review_pk VARCHAR NOT NULL,
            parent_asin VARCHAR NOT NULL,
            asin VARCHAR,
            user_id VARCHAR,
            timestamp INTEGER NOT NULL,
            rating DOUBLE NOT NULL,
            verified_purchase BOOLEAN,
            helpful_vote INTEGER,
            review_title VARCHAR,
            review_text VARCHAR NOT NULL,
            main_category VARCHAR,
            product_title VARCHAR
        """
        self.db.create_table_if_not_exists(self.SELECTED_REVIEWS, schema)
    
    def create_meta_context_table(self):
        """创建meta_context表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            parent_asin VARCHAR NOT NULL,
            product_title VARCHAR NOT NULL,
            main_category VARCHAR,
            features_short VARCHAR,
            description_short VARCHAR,
            details_short VARCHAR,
            context_version VARCHAR NOT NULL
        """
        self.db.create_table_if_not_exists(self.META_CONTEXT, schema)
    
    def create_normalized_reviews_table(self):
        """创建normalized_reviews表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            review_pk VARCHAR NOT NULL,
            parent_asin VARCHAR NOT NULL,
            timestamp INTEGER NOT NULL,
            rating DOUBLE NOT NULL,
            clean_text VARCHAR NOT NULL,
            cleaning_flags VARCHAR
        """
        self.db.create_table_if_not_exists(self.NORMALIZED_REVIEWS, schema)
    
    def create_review_sentences_table(self):
        """创建review_sentences表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            sentence_id VARCHAR NOT NULL,
            review_pk VARCHAR NOT NULL,
            parent_asin VARCHAR NOT NULL,
            timestamp INTEGER NOT NULL,
            rating DOUBLE NOT NULL,
            verified_purchase BOOLEAN,
            helpful_vote INTEGER,
            sentence_index INTEGER NOT NULL,
            target_sentence VARCHAR NOT NULL,
            prev_sentence VARCHAR,
            next_sentence VARCHAR,
            context_text VARCHAR NOT NULL
        """
        self.db.create_table_if_not_exists(self.REVIEW_SENTENCES, schema)
    
    def create_opinion_candidates_table(self):
        """创建opinion_candidates表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            sentence_id VARCHAR NOT NULL,
            is_candidate BOOLEAN NOT NULL,
            filter_reason VARCHAR,
            priority_weight DOUBLE
        """
        self.db.create_table_if_not_exists(self.OPINION_CANDIDATES, schema)
    
    def create_aspect_sentiment_raw_table(self):
        """创建aspect_sentiment_raw表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            llm_model VARCHAR NOT NULL,
            prompt_version VARCHAR NOT NULL,
            sentence_id VARCHAR NOT NULL,
            parse_status VARCHAR NOT NULL,
            retry_count INTEGER NOT NULL,
            error_type VARCHAR,
            llm_output JSON
        """
        self.db.create_table_if_not_exists(self.ASPECT_SENTIMENT_RAW, schema)
    
    def create_aspect_sentiment_valid_table(self):
        """创建aspect_sentiment_valid表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            sentence_id VARCHAR NOT NULL,
            review_pk VARCHAR NOT NULL,
            parent_asin VARCHAR NOT NULL,
            timestamp INTEGER NOT NULL,
            rating DOUBLE NOT NULL,
            aspect_raw VARCHAR NOT NULL,
            aspect_norm VARCHAR NOT NULL,
            sentiment VARCHAR NOT NULL,
            sentiment_score DOUBLE,
            issue_raw VARCHAR NOT NULL,
            issue_norm VARCHAR NOT NULL,
            evidence_text VARCHAR NOT NULL,
            validity_label VARCHAR NOT NULL,
            quality_flags VARCHAR
        """
        self.db.create_table_if_not_exists(self.ASPECT_SENTIMENT_VALID, schema)
    
    def create_extraction_issues_table(self):
        """创建extraction_issues表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            sentence_id VARCHAR NOT NULL,
            issue_type VARCHAR NOT NULL,
            details VARCHAR,
            needs_recheck BOOLEAN
        """
        self.db.create_table_if_not_exists(self.EXTRACTION_ISSUES, schema)
    
    def create_issue_clusters_table(self, embedding_dim: Optional[int] = None):
        """
        创建issue_clusters表
        
        Args:
            embedding_dim: embedding向量维度（可选，如果提供则添加向量字段）
        """
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            embedding_model VARCHAR NOT NULL,
            clustering_config_id VARCHAR NOT NULL,
            aspect_norm VARCHAR NOT NULL,
            cluster_id VARCHAR NOT NULL,
            sentence_id VARCHAR NOT NULL,
            cluster_key_text VARCHAR NOT NULL,
            issue_norm VARCHAR NOT NULL,
            sentiment VARCHAR NOT NULL,
            is_noise BOOLEAN
        """
        # 如果指定了embedding维度，添加向量字段
        if embedding_dim is not None:
            schema += f",\n            cluster_embedding FLOAT[{embedding_dim}]"
        
        self.db.create_table_if_not_exists(self.ISSUE_CLUSTERS, schema)
        
        # 如果表已存在且需要添加向量字段，尝试添加列（如果不存在）
        if embedding_dim is not None:
            try:
                # 直接尝试添加列，如果列已存在会抛出异常，捕获并忽略
                alter_query = f"ALTER TABLE {self.ISSUE_CLUSTERS} ADD COLUMN cluster_embedding FLOAT[{embedding_dim}]"
                self.db.execute_write(alter_query)
            except Exception as e:
                # 如果添加列失败（可能是列已存在或其他原因），记录调试信息但不中断
                from ..utils.logger import get_logger
                logger = get_logger(__name__)
                error_msg = str(e).lower()
                if "already exists" in error_msg or "duplicate" in error_msg:
                    logger.debug(f"cluster_embedding列已存在，跳过添加")
                else:
                    logger.warning(f"添加cluster_embedding列时出现警告: {e}")
    
    def create_cluster_stats_table(self):
        """创建cluster_stats表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            aspect_norm VARCHAR NOT NULL,
            cluster_id VARCHAR NOT NULL,
            cluster_size INTEGER NOT NULL,
            neg_ratio DOUBLE NOT NULL,
            intra_cluster_distance DOUBLE,
            inter_cluster_distance DOUBLE,
            separation_ratio DOUBLE,
            cohesion DOUBLE,
            cluster_confidence DOUBLE,
            sentiment_consistency DOUBLE,
            recent_trend JSON,
            top_terms JSON,
            representative_sentence_ids JSON NOT NULL
        """
        self.db.create_table_if_not_exists(self.CLUSTER_STATS, schema)
    
    def create_cluster_reports_table(self):
        """创建cluster_reports表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            llm_model VARCHAR NOT NULL,
            prompt_version VARCHAR NOT NULL,
            aspect_norm VARCHAR NOT NULL,
            cluster_id VARCHAR NOT NULL,
            cluster_name VARCHAR NOT NULL,
            summary VARCHAR NOT NULL,
            priority VARCHAR NOT NULL,
            evidence_items JSON NOT NULL,
            action_items JSON NOT NULL,
            risks_and_assumptions JSON,
            confidence DOUBLE
        """
        self.db.create_table_if_not_exists(self.CLUSTER_REPORTS, schema)
    
    def create_evaluation_metrics_table(self):
        """创建evaluation_metrics表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            metric_name VARCHAR NOT NULL,
            metric_value DOUBLE NOT NULL,
            metric_scope VARCHAR,
            notes VARCHAR
        """
        self.db.create_table_if_not_exists(self.EVALUATION_METRICS, schema)
    
    def create_run_log_table(self):
        """创建run_log表"""
        schema = """
            run_id VARCHAR NOT NULL,
            step_name VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            input_rows INTEGER,
            output_rows INTEGER,
            error_rows INTEGER,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            message VARCHAR
        """
        self.db.create_table_if_not_exists(self.RUN_LOG, schema)
    
    def create_aspect_sentiment_raw_traditional_table(self):
        """创建aspect_sentiment_raw_traditional表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            sentence_id VARCHAR NOT NULL,
            extract_method VARCHAR NOT NULL,
            aspect_raw VARCHAR NOT NULL,
            issue_raw VARCHAR NOT NULL,
            sentiment VARCHAR NOT NULL,
            sentiment_score DOUBLE,
            evidence_text VARCHAR NOT NULL,
            debug_features JSON
        """
        self.db.create_table_if_not_exists(self.ASPECT_SENTIMENT_RAW_TRADITIONAL, schema)
    
    def create_aspect_sentiment_valid_traditional_table(self):
        """创建aspect_sentiment_valid_traditional表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            sentence_id VARCHAR NOT NULL,
            review_pk VARCHAR NOT NULL,
            parent_asin VARCHAR NOT NULL,
            timestamp INTEGER NOT NULL,
            rating DOUBLE NOT NULL,
            aspect_raw VARCHAR NOT NULL,
            aspect_norm VARCHAR NOT NULL,
            sentiment VARCHAR NOT NULL,
            sentiment_score DOUBLE,
            issue_raw VARCHAR NOT NULL,
            issue_norm VARCHAR NOT NULL,
            evidence_text VARCHAR NOT NULL,
            validity_label VARCHAR NOT NULL,
            quality_flags VARCHAR
        """
        self.db.create_table_if_not_exists(self.ASPECT_SENTIMENT_VALID_TRADITIONAL, schema)
    
    def create_extraction_issues_traditional_table(self):
        """创建extraction_issues_traditional表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            sentence_id VARCHAR NOT NULL,
            issue_type VARCHAR NOT NULL,
            details VARCHAR,
            needs_recheck BOOLEAN
        """
        self.db.create_table_if_not_exists(self.EXTRACTION_ISSUES_TRADITIONAL, schema)
    
    def create_issue_clusters_traditional_table(self, embedding_dim: Optional[int] = None):
        """
        创建issue_clusters_traditional表
        
        Args:
            embedding_dim: embedding向量维度（可选，如果提供则添加向量字段）
        """
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            embedding_model VARCHAR NOT NULL,
            clustering_config_id VARCHAR NOT NULL,
            aspect_norm VARCHAR NOT NULL,
            cluster_id VARCHAR NOT NULL,
            sentence_id VARCHAR NOT NULL,
            cluster_key_text VARCHAR NOT NULL,
            issue_norm VARCHAR NOT NULL,
            sentiment VARCHAR NOT NULL,
            is_noise BOOLEAN
        """
        # 如果指定了embedding维度，添加向量字段
        if embedding_dim is not None:
            schema += f",\n            cluster_embedding FLOAT[{embedding_dim}]"
        
        self.db.create_table_if_not_exists(self.ISSUE_CLUSTERS_TRADITIONAL, schema)
        
        # 如果表已存在且需要添加向量字段，尝试添加列（如果不存在）
        if embedding_dim is not None:
            try:
                # 直接尝试添加列，如果列已存在会抛出异常，捕获并忽略
                alter_query = f"ALTER TABLE {self.ISSUE_CLUSTERS_TRADITIONAL} ADD COLUMN cluster_embedding FLOAT[{embedding_dim}]"
                self.db.execute_write(alter_query)
            except Exception as e:
                # 如果添加列失败（可能是列已存在或其他原因），记录调试信息但不中断
                from ..utils.logger import get_logger
                logger = get_logger(__name__)
                error_msg = str(e).lower()
                if "already exists" in error_msg or "duplicate" in error_msg:
                    logger.debug(f"cluster_embedding列已存在，跳过添加")
                else:
                    logger.warning(f"添加cluster_embedding列时出现警告: {e}")
    
    def create_cluster_stats_traditional_table(self):
        """创建cluster_stats_traditional表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            aspect_norm VARCHAR NOT NULL,
            cluster_id VARCHAR NOT NULL,
            cluster_size INTEGER NOT NULL,
            neg_ratio DOUBLE NOT NULL,
            intra_cluster_distance DOUBLE,
            inter_cluster_distance DOUBLE,
            separation_ratio DOUBLE,
            cohesion DOUBLE,
            cluster_confidence DOUBLE,
            sentiment_consistency DOUBLE,
            recent_trend JSON,
            top_terms JSON,
            representative_sentence_ids JSON NOT NULL
        """
        self.db.create_table_if_not_exists(self.CLUSTER_STATS_TRADITIONAL, schema)
    
    def create_cluster_reports_traditional_table(self):
        """创建cluster_reports_traditional表"""
        schema = """
            run_id VARCHAR NOT NULL,
            pipeline_version VARCHAR NOT NULL,
            data_slice_id VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            method_version VARCHAR NOT NULL,
            aspect_norm VARCHAR NOT NULL,
            cluster_id VARCHAR NOT NULL,
            cluster_name VARCHAR NOT NULL,
            summary VARCHAR NOT NULL,
            priority VARCHAR NOT NULL,
            priority_rationale VARCHAR,
            evidence_items JSON NOT NULL,
            action_items JSON NOT NULL,
            confidence DOUBLE
        """
        self.db.create_table_if_not_exists(self.CLUSTER_REPORTS_TRADITIONAL, schema)
    
    def get_run_view_query(self, table_name: str, run_id: str) -> str:
        """生成run_id过滤视图查询"""
        return f"SELECT * FROM {table_name} WHERE run_id = '{run_id}'"

