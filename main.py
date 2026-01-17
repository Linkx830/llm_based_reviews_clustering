# 新增或修改我时需要修改这个文件夹中的README.md文件
"""主程序入口"""
import argparse
from pathlib import Path
import yaml
import logging
from dotenv import load_dotenv
import os

from src.app import Orchestrator
from src.pipelines import Pipeline, PipelineStep
from src.agents import (
    DataSelectorAgent,
    MetaContextAgent,
    PreprocessAgent,
    SentenceBuilderAgent,
    OpinionCandidateFilterAgent,
    AspectSentimentExtractorAgent,
    ExtractionJudgeAgent,
    IssueClusterAgent,
    ClusterInsightAgent,
    ReportAssemblerAgent,
    EvaluationAgent,
    TraditionalAspectSentimentExtractorAgent,
    TraditionalExtractionJudgeAgent,
    TraditionalClusterInsightAgent
)
from src.models import LLMWrapper
from src.storage.version_fields import VersionFields
from src.utils.config_loader import load_prompt_template
from src.utils.logger import setup_logger


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_llm_wrapper(
    config: dict,
    task_type: str = "default",
    env_prefix: str = ""
) -> LLMWrapper:
    """
    创建LLM wrapper
    
    Args:
        config: 配置字典
        task_type: 任务类型（"extraction" 或 "insight" 或 "default"）
        env_prefix: 环境变量前缀（如 "EXTRACTION_" 或 "INSIGHT_"）
    
    Returns:
        LLMWrapper实例
    """
    models_config = config.get("models", {})
    
    # 优先使用任务特定的配置
    if task_type == "extraction" and "extraction_llm" in models_config:
        llm_config = models_config["extraction_llm"]
        provider = os.getenv(f"{env_prefix}LLM_PROVIDER", llm_config.get("provider", "ollama"))
        model = os.getenv(f"{env_prefix}LLM_MODEL", llm_config.get("model", "qwen3:8b"))
        base_url = os.getenv(f"{env_prefix}LLM_BASE_URL", llm_config.get("base_url"))
        temperature = llm_config.get("temperature", 0.0)
        api_key = os.getenv(f"{env_prefix}API_KEY", llm_config.get("api_key"))
        enable_reasoning = llm_config.get("enable_reasoning", False)
    elif task_type == "insight" and "insight_llm" in models_config:
        llm_config = models_config["insight_llm"]
        provider = os.getenv(f"{env_prefix}LLM_PROVIDER", llm_config.get("provider", "ollama"))
        model = os.getenv(f"{env_prefix}LLM_MODEL", llm_config.get("model", "qwen3:8b"))
        base_url = os.getenv(f"{env_prefix}LLM_BASE_URL", llm_config.get("base_url"))
        temperature = llm_config.get("temperature", 0.3)
        api_key = os.getenv(f"{env_prefix}API_KEY", llm_config.get("api_key"))
        enable_reasoning = llm_config.get("enable_reasoning", False)
    elif task_type == "judge" and "judge_llm" in models_config:
        llm_config = models_config["judge_llm"]
        provider = os.getenv(f"{env_prefix}LLM_PROVIDER", llm_config.get("provider", "ollama"))
        model = os.getenv(f"{env_prefix}LLM_MODEL", llm_config.get("model", "qwen3:8b"))
        base_url = os.getenv(f"{env_prefix}LLM_BASE_URL", llm_config.get("base_url"))
        temperature = llm_config.get("temperature", 0.0)
        api_key = os.getenv(f"{env_prefix}API_KEY", llm_config.get("api_key"))
        enable_reasoning = llm_config.get("enable_reasoning", False)
    else:
        # 使用默认配置（向后兼容）
        provider = os.getenv("LLM_PROVIDER", models_config.get("llm_provider", "ollama"))
        model = os.getenv("LLM_MODEL", models_config.get("llm_model", "qwen3:8b"))
        base_url = os.getenv("LLM_BASE_URL", models_config.get("llm_base_url"))
        temperature = models_config.get("temperature", 0.0)
        api_key = os.getenv("API_KEY", models_config.get("api_key"))
        enable_reasoning = models_config.get("enable_reasoning", False)
    
    return LLMWrapper(
        provider=provider,
        model_name=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        enable_reasoning=enable_reasoning
    )


def create_pipeline(config: dict) -> Pipeline:
    """创建流水线"""
    # 加载环境变量
    load_dotenv()
    
    # 为不同任务创建不同的LLM wrapper
    extraction_llm_wrapper = create_llm_wrapper(config, task_type="extraction")
    insight_llm_wrapper = create_llm_wrapper(config, task_type="insight")
    
    # 创建embedding工具（用于ExtractionJudgeAgent的语义匹配和ClusterInsightAgent的sentence_id匹配，可选）
    embedding_tool = None
    use_semantic_matching = config.get("judge", {}).get("use_semantic_matching", False)
    # 即使judge不使用语义匹配，insight步骤也可能需要embedding工具来匹配sentence_id
    # 所以总是创建embedding_tool（如果配置了embedding_model）
    if config["models"].get("embedding_model"):
        from src.tools.embedding_tool import EmbeddingTool
        embedding_model = config["models"].get("embedding_model", "all-MiniLM-L6-v2")
        embedding_base_url = config["models"].get("embedding_base_url") or config["models"].get("llm_base_url")
        embedding_max_workers = config["models"].get("embedding_max_workers", 4)
        embedding_batch_size = config["models"].get("embedding_batch_size", 32)
        embedding_tool = EmbeddingTool(
            model_name=embedding_model,
            base_url=embedding_base_url,
            max_workers=embedding_max_workers,
            batch_size=embedding_batch_size
        )
    
    # 创建LLM wrapper（用于ExtractionJudgeAgent的LLM辅助匹配，可选）
    judge_llm_wrapper = None
    use_llm_matching = config.get("judge", {}).get("use_llm_matching", False)
    if use_llm_matching:
        # 可以使用较小的模型进行归一化任务，也可以复用extraction_llm
        models_config = config.get("models", {})
        if "judge_llm" in models_config:
            # 如果配置了专门的judge LLM，使用它
            judge_llm_wrapper = create_llm_wrapper(config, task_type="judge")
        else:
            # 否则复用extraction_llm（通常已经配置好了）
            judge_llm_wrapper = extraction_llm_wrapper
    
    # 定义步骤
    steps = [
        PipelineStep("data_selector", DataSelectorAgent, dependencies=[]),
        PipelineStep("meta_context", MetaContextAgent, dependencies=["data_selector"]),
        PipelineStep("preprocess", PreprocessAgent, dependencies=["data_selector"]),
        PipelineStep("sentence_builder", SentenceBuilderAgent, dependencies=["preprocess"]),
        PipelineStep("opinion_filter", OpinionCandidateFilterAgent, dependencies=["sentence_builder"]),
        PipelineStep(
            "extraction",
            AspectSentimentExtractorAgent,
            dependencies=["opinion_filter", "meta_context"],
            config={"llm_wrapper": extraction_llm_wrapper, "prompt_version": config["prompts"]["extraction"]}
        ),
        PipelineStep(
            "judge", 
            ExtractionJudgeAgent, 
            dependencies=["extraction"],
            config={
                "embedding_tool": embedding_tool,
                "use_semantic_matching": use_semantic_matching,
                "semantic_threshold": config.get("judge", {}).get("semantic_threshold", 0.7),
                "llm_wrapper": judge_llm_wrapper,
                "use_llm_matching": use_llm_matching,
                "llm_confidence_threshold": config.get("judge", {}).get("llm_confidence_threshold", 0.5)
            }
        ),
        PipelineStep(
            "clustering",
            IssueClusterAgent,
            dependencies=["judge"],
            config={
                "embedding_model": config["models"]["embedding_model"],
                "embedding_base_url": config["models"].get("embedding_base_url") or config["models"].get("llm_base_url"),
                "embedding_max_workers": config["models"].get("embedding_max_workers", 4),
                "embedding_batch_size": config["models"].get("embedding_batch_size", 32),
                "embedding_mrl_dimensions": config["models"].get("embedding_mrl_dimensions"),
                "clustering_config": config["clustering"],
                # 阶段D：Reranker参数
                "use_reranker": config["clustering"].get("use_reranker", False),
                "reranker_model": config["clustering"].get("reranker_model"),
                "reranker_base_url": config["clustering"].get("reranker_base_url") or config["models"].get("embedding_base_url"),
                # 提供extraction_llm_wrapper作为回退选项（当Ollama reranker失败时自动使用）
                "reranker_llm_wrapper": extraction_llm_wrapper if config["clustering"].get("use_reranker", False) else None,
                "reranker_top_k": config["clustering"].get("reranker_top_k", 50),
                "reranker_score_threshold": config["clustering"].get("reranker_score_threshold", 0.6),
                "reranker_max_workers": config["clustering"].get("reranker_max_workers", 4),
                # 阶段E：噪点簇和小簇后处理参数
                "min_cluster_size": config["clustering"].get("min_cluster_size", 2),
                "noise_adsorption_threshold": config["clustering"].get("noise_adsorption_threshold", 0.7),
                "small_cluster_merge_threshold": config["clustering"].get("small_cluster_merge_threshold", 0.75),
                # 两阶段聚类控制参数
                "enable_two_stage_clustering": config["clustering"].get("enable_two_stage_clustering", True),
                "two_stage_threshold": config["clustering"].get("two_stage_threshold", 100.0)
            }
        ),
        PipelineStep(
            "insight",
            ClusterInsightAgent,
            dependencies=["clustering"],
            config={
                "llm_wrapper": insight_llm_wrapper,
                "prompt_version": config["prompts"]["insight"],
                "embedding_tool": embedding_tool  # 可选，用于向量相似度匹配sentence_id
            }
        ),
        PipelineStep("evaluation", EvaluationAgent, dependencies=["judge", "clustering"]),
        PipelineStep(
            "report",
            ReportAssemblerAgent,
            dependencies=["insight", "evaluation"],
            config={}
        )
    ]
    
    return Pipeline(steps)


def create_traditional_pipeline(config: dict) -> Pipeline:
    """创建传统baseline流水线（不使用LLM）"""
    # 加载环境变量
    load_dotenv()
    
    # 创建embedding工具（传统方法仍然使用embedding做聚类）
    embedding_tool = None
    if config["models"].get("embedding_model"):
        from src.tools.embedding_tool import EmbeddingTool
        embedding_model = config["models"].get("embedding_model", "all-MiniLM-L6-v2")
        embedding_base_url = config["models"].get("embedding_base_url") or config["models"].get("llm_base_url")
        embedding_max_workers = config["models"].get("embedding_max_workers", 4)
        embedding_batch_size = config["models"].get("embedding_batch_size", 32)
        embedding_tool = EmbeddingTool(
            model_name=embedding_model,
            base_url=embedding_base_url,
            max_workers=embedding_max_workers,
            batch_size=embedding_batch_size
        )
    
    # 获取传统方法配置
    traditional_config = config.get("traditional", {})
    
    # 定义步骤（与传统方法对齐）
    steps = [
        PipelineStep("data_selector", DataSelectorAgent, dependencies=[]),
        PipelineStep("meta_context", MetaContextAgent, dependencies=["data_selector"]),
        PipelineStep("preprocess", PreprocessAgent, dependencies=["data_selector"]),
        PipelineStep("sentence_builder", SentenceBuilderAgent, dependencies=["preprocess"]),
        PipelineStep("opinion_filter", OpinionCandidateFilterAgent, dependencies=["sentence_builder"]),
        PipelineStep(
            "extraction",
            TraditionalAspectSentimentExtractorAgent,
            dependencies=["opinion_filter", "meta_context"],
            config={
                "extract_method": traditional_config.get("extraction", {}).get("method", "LEXICON_RULE"),
                "aspect_seed_lexicon": traditional_config.get("extraction", {}).get("aspect_seed_lexicon"),
                "use_meta_context": traditional_config.get("extraction", {}).get("use_meta_context", True),
                "max_candidates_per_sentence": traditional_config.get("extraction", {}).get("max_candidates_per_sentence", 5)
            }
        ),
        PipelineStep(
            "judge", 
            TraditionalExtractionJudgeAgent, 
            dependencies=["extraction"],
            config={
                "embedding_tool": embedding_tool,
                "use_semantic_matching": traditional_config.get("judge", {}).get("use_semantic_matching", False),
                "semantic_threshold": traditional_config.get("judge", {}).get("semantic_threshold", 0.7)
            }
        ),
        PipelineStep(
            "clustering",
            IssueClusterAgent,
            dependencies=["judge"],
            config={
                "embedding_model": config["models"]["embedding_model"],
                "embedding_base_url": config["models"].get("embedding_base_url") or config["models"].get("llm_base_url"),
                "embedding_max_workers": config["models"].get("embedding_max_workers", 4),
                "embedding_batch_size": config["models"].get("embedding_batch_size", 32),
                "embedding_mrl_dimensions": config["models"].get("embedding_mrl_dimensions"),
                "clustering_config": config["clustering"],
                # 阶段D：Reranker参数（可选，为公平对比建议同样开启/关闭）
                "use_reranker": config["clustering"].get("use_reranker", False),
                "reranker_model": config["clustering"].get("reranker_model"),
                "reranker_base_url": config["clustering"].get("reranker_base_url") or config["models"].get("embedding_base_url"),
                "reranker_llm_wrapper": None,  # 传统方法不使用LLM reranker
                "reranker_top_k": config["clustering"].get("reranker_top_k", 50),
                "reranker_score_threshold": config["clustering"].get("reranker_score_threshold", 0.6),
                "reranker_max_workers": config["clustering"].get("reranker_max_workers", 4),
                # 阶段E：噪点簇和小簇后处理参数
                "min_cluster_size": config["clustering"].get("min_cluster_size", 2),
                "noise_adsorption_threshold": config["clustering"].get("noise_adsorption_threshold", 0.7),
                "small_cluster_merge_threshold": config["clustering"].get("small_cluster_merge_threshold", 0.75),
                # 两阶段聚类控制参数
                "enable_two_stage_clustering": config["clustering"].get("enable_two_stage_clustering", True),
                "two_stage_threshold": config["clustering"].get("two_stage_threshold", 100.0),
                # 传统模式
                "use_traditional_mode": True
            }
        ),
        PipelineStep(
            "insight",
            TraditionalClusterInsightAgent,
            dependencies=["clustering"],
            config={
                "method_version": traditional_config.get("insight", {}).get("method_version", "v1.0"),
                "evidence_count": traditional_config.get("insight", {}).get("evidence_count", 5),
                "action_rule_table": traditional_config.get("insight", {}).get("action_rule_table")
            }
        ),
        PipelineStep("evaluation", EvaluationAgent, dependencies=["judge", "clustering"]),
        PipelineStep(
            "report",
            ReportAssemblerAgent,
            dependencies=["insight", "evaluation"],
            config={}
        )
    ]
    
    return Pipeline(steps)


def main():
    parser = argparse.ArgumentParser(description="运行数据挖掘流水线")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--run-id", type=str, help="运行ID（可选）")
    parser.add_argument("--resume-from", type=str, help="从指定步骤恢复")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置控制台日志（运行流水线时会在runs目录中创建文件日志）
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logger("root", level=log_level)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("开始运行数据挖掘流水线")
    logger.info("=" * 60)
    
    # 加载配置
    config = load_config(Path(args.config))
    logger.info(f"加载配置文件: {args.config}")
    
    # 创建Orchestrator
    db_path = os.getenv("DUCKDB_PATH", "data/duckDB/amazon_appliance.duckdb")
    orchestrator = Orchestrator(
        db_path=db_path,
        run_id=args.run_id or config["run"].get("run_id"),
        pipeline_version=config["run"]["pipeline_version"]
    )
    
    try:
        # 创建流水线（根据配置选择传统或LLM方法）
        use_traditional = config.get("run", {}).get("use_traditional", False)
        if use_traditional:
            logger.info("使用传统baseline流水线（无LLM）")
            pipeline = create_traditional_pipeline(config)
        else:
            logger.info("使用LLM流水线")
            pipeline = create_pipeline(config)
        
        # 准备配置
        run_config = {
            "data_selector": config["data_slice"],
            "meta_context": {},
            "preprocess": {},
            "sentence_builder": {},
            "opinion_filter": {},
            "extraction": {},
            "judge": {},
            "clustering": {},
            "insight": {},
            "report": {},
            "evaluation": {}
        }
        
        # 运行流水线
        config_path = Path(args.config)
        result = orchestrator.run_pipeline(
            pipeline=pipeline,
            config=run_config,
            resume_from=args.resume_from,
            full_config=config,  # 传递完整配置
            config_file_path=config_path  # 传递配置文件路径
        )
        
        logger.info("=" * 60)
        logger.info(f"运行完成！Run ID: {result['run_id']}")
        logger.info(f"输出目录: {result['output_dir']}")
        if 'log_file' in result:
            logger.info(f"日志文件: {result['log_file']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"运行失败: {str(e)}", exc_info=True)
        raise
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()

