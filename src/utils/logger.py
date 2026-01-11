# 新增或修改我时需要修改这个文件夹中的README.md文件
"""日志工具"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        format_string: 格式字符串（可选）
    
    Returns:
        配置好的Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 检查是否已有控制台handler
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) 
                      for h in logger.handlers)
    
    # 控制台输出（如果没有则添加）
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出（如果指定且该文件handler不存在）
    if log_file:
        # 检查是否已有相同文件的handler
        log_file_abs = str(log_file.absolute())
        has_file = any(
            isinstance(h, logging.FileHandler) and 
            hasattr(h, 'baseFilename') and 
            str(Path(h.baseFilename).absolute()) == log_file_abs
            for h in logger.handlers
        )
        if not has_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)

