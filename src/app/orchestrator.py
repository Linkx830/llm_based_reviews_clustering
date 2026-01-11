# 新增或修改我时需要修改这个文件夹中的README.md文件
"""Orchestrator - 编排器主控"""
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging
import yaml
import shutil
from ..storage.connection import DuckDBConnection
from ..storage.table_manager import TableManager
from ..storage.version_fields import VersionFields
from ..tools.logging_tool import LoggingTool
from ..pipelines.pipeline import Pipeline, PipelineStep
from ..utils.logger import setup_logger


class Orchestrator:
    """
    Orchestrator - 编排器
    
    职责：
    - 控制执行顺序与依赖关系
    - 统一写入DuckDB（单写者）
    - 执行断点续跑与版本一致性检查
    - 记录运行日志
    """
    
    def __init__(
        self,
        db_path: str,
        run_id: Optional[str] = None,
        pipeline_version: str = "v1.0"
    ):
        """
        Args:
            db_path: DuckDB文件路径
            run_id: 运行ID（如果为None则自动生成）
            pipeline_version: 管道版本
        """
        self.db_conn = DuckDBConnection.get_instance(db_path)
        self.db_conn.connect()
        self.table_manager = TableManager(self.db_conn)
        self.logger = LoggingTool(self.db_conn)
        
        # 初始化表
        self.table_manager.create_all_tables()
        
        # 版本信息
        self.pipeline_version = pipeline_version
        self.run_id = run_id or VersionFields.generate_run_id()
        self.data_slice_id = None  # 将在DataSelectorAgent中设置
    
    def run_pipeline(
        self,
        pipeline: Pipeline,
        config: Dict[str, Any],
        output_dir: Optional[Path] = None,
        resume_from: Optional[str] = None,
        full_config: Optional[Dict[str, Any]] = None,
        config_file_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        运行流水线
        
        Args:
            pipeline: Pipeline定义
            config: 配置字典（处理参数）
            output_dir: 输出目录
            resume_from: 从哪个步骤恢复（断点续跑）
            full_config: 完整配置字典（用于保存到config文件夹）
            config_file_path: 原始配置文件路径（用于复制到config文件夹）
        
        Returns:
            运行结果
        """
        if output_dir is None:
            output_dir = Path("outputs/runs") / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建config文件夹并保存运行参数
        config_dir = output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        self._save_run_config(config_dir, full_config, config_file_path)
        
        # 设置日志文件路径
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "orchestrator.log"
        
        # 配置日志记录器，将日志写入文件
        setup_logger(
            name="orchestrator",
            log_file=log_file,
            level=logging.INFO
        )
        setup_logger(
            name="root",
            log_file=log_file,
            level=logging.INFO
        )
        
        # 获取日志记录器并记录运行开始
        logger = logging.getLogger("orchestrator")
        logger.info("=" * 60)
        logger.info(f"开始运行流水线 - Run ID: {self.run_id}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"日志文件: {log_file}")
        logger.info("=" * 60)
        
        execution_order = pipeline.get_execution_order()
        completed_steps = set()
        
        # 如果指定了恢复点，跳过已完成的步骤
        if resume_from:
            resume_idx = execution_order.index(resume_from) if resume_from in execution_order else 0
            completed_steps = set(execution_order[:resume_idx])
        
        results = {}
        
        for step_name in execution_order:
            if step_name in completed_steps:
                continue
            
            step = pipeline.get_step(step_name)
            if not step:
                continue
            
            # 记录开始
            self.logger.log_step_start(self.run_id, step_name)
            
            try:
                # 创建Agent实例（只使用step.config中的初始化参数）
                agent = step.agent_class(
                    self.db_conn,
                    self.run_id,
                    self.pipeline_version,
                    self.data_slice_id or "default",
                    **step.config
                )
                
                # 执行处理（使用config中的处理参数）
                process_params = config.get(step_name, {}).copy()
                
                # 特殊处理：report步骤需要output_dir
                if step_name == "report":
                    process_params["output_dir"] = output_dir
                
                result = agent.process(**process_params)
                results[step_name] = result
                
                # 更新data_slice_id（如果DataSelectorAgent设置了）
                if step_name == "data_selector" and "data_slice_id" in result:
                    self.data_slice_id = result["data_slice_id"]
                
                # 记录成功
                self.logger.log_step_success(
                    self.run_id,
                    step_name,
                    output_rows=result.get("processed_count") or result.get("sentence_count") or 0
                )
                completed_steps.add(step_name)
                
            except Exception as e:
                # 记录失败
                self.logger.log_step_failed(
                    self.run_id,
                    step_name,
                    message=str(e)
                )
                raise
        
        # 记录运行完成
        logger = logging.getLogger("orchestrator")
        logger.info("=" * 60)
        logger.info(f"流水线运行完成 - Run ID: {self.run_id}")
        logger.info("=" * 60)
        
        return {
            "run_id": self.run_id,
            "status": "success",
            "results": results,
            "output_dir": str(output_dir),
            "log_file": str(log_file)
        }
    
    def _save_run_config(
        self,
        config_dir: Path,
        full_config: Optional[Dict[str, Any]],
        config_file_path: Optional[Path]
    ):
        """
        保存运行配置到config文件夹
        
        Args:
            config_dir: config文件夹路径
            full_config: 完整配置字典
            config_file_path: 原始配置文件路径
        """
        # 1. 如果提供了原始配置文件路径，复制原始配置文件
        if config_file_path and config_file_path.exists():
            dest_config_file = config_dir / config_file_path.name
            shutil.copy2(config_file_path, dest_config_file)
        
        # 2. 保存完整配置字典（包含运行时信息）
        if full_config is None:
            full_config = {}
        
        # 清理配置字典，移除不可序列化的对象（如函数、类实例等）
        def clean_config(obj):
            """递归清理配置对象，移除不可序列化的内容"""
            if isinstance(obj, dict):
                return {k: clean_config(v) for k, v in obj.items() if not k.startswith('_')}
            elif isinstance(obj, (list, tuple)):
                return [clean_config(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, Path):
                return str(obj)
            else:
                # 对于其他类型，尝试转换为字符串
                try:
                    return str(obj)
                except:
                    return None
        
        cleaned_config = clean_config(full_config)
        
        # 添加运行时信息
        run_config = {
            "run_info": {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id or "default",
                "created_at": datetime.now().isoformat(),
                "config_file": str(config_file_path) if config_file_path else None
            },
            **cleaned_config
        }
        
        # 保存为YAML文件
        config_yaml_path = config_dir / "run_config.yaml"
        with open(config_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(run_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        # 3. 保存环境变量信息（不包含敏感信息）
        import sys
        env_info = {
            "DUCKDB_PATH": str(self.db_conn.db_path.resolve()) if hasattr(self.db_conn, 'db_path') else None,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
        
        env_yaml_path = config_dir / "environment.yaml"
        with open(env_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(env_info, f, allow_unicode=True, default_flow_style=False)
    
    def close(self):
        """关闭连接"""
        self.db_conn.close()

