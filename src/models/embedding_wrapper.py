# 新增或修改我时需要修改这个文件夹中的README.md文件
"""Embedding封装"""
from ..tools.embedding_tool import EmbeddingTool


class EmbeddingWrapper:
    """
    Embedding封装（代理到EmbeddingTool）
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.tool = EmbeddingTool(model_name)
    
    def encode(self, texts: list, batch_size: int = 32):
        """编码文本列表"""
        return self.tool.encode(texts, batch_size)

