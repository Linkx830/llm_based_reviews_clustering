# 新增或修改我时需要修改这个文件夹中的README.md文件
"""Embedding向量化工具"""
from typing import List, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import requests
except ImportError:
    requests = None


class EmbeddingTool:
    """
    Embedding向量化工具
    
    职责：
    - 文本向量化
    - 支持sentence-transformers模型和Ollama embedding模型
    - 支持批量处理和并发请求加速
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        base_url: Optional[str] = None,
        max_workers: int = 4,
        batch_size: int = 32,
        mrl_dimensions: Optional[int] = None
    ):
        """
        Args:
            model_name: 模型名称
                - sentence-transformers模型名称（如 "all-MiniLM-L6-v2"）
                - Ollama模型名称（如 "qwen3-embedding:4b"）
            base_url: Ollama API地址（仅当使用Ollama模型时需要）
            max_workers: Ollama并发请求数（默认4，可根据服务器性能调整）
            batch_size: sentence-transformers批处理大小（默认32）
            mrl_dimensions: MRL维度裁剪（可选，如768/512/256，仅支持支持MRL的模型如Qwen3-Embedding）
        """
        self.model_name = model_name
        self.base_url = base_url or "http://localhost:11434"
        self.model: Optional[SentenceTransformer] = None
        self.use_ollama = self._is_ollama_model(model_name)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.mrl_dimensions = mrl_dimensions
    
    def _is_ollama_model(self, model_name: str) -> bool:
        """判断是否为Ollama模型"""
        ollama_keywords = ["embedding", "ollama"]
        return any(keyword in model_name.lower() for keyword in ollama_keywords)
    
    def load_model(self):
        """加载模型"""
        if not self.use_ollama:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required for non-Ollama models")
            if self.model is None:
                self.model = SentenceTransformer(self.model_name)
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        instruction: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        将文本列表编码为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
                - sentence-transformers: 批处理大小（默认使用初始化时的batch_size）
                - Ollama: 并发请求数（默认使用初始化时的max_workers）
            instruction: 可选的instruction文本（用于instruction-aware embedding）
                格式：如果提供，会按照 "Instruct: {instruction}\\nQuery: {text}" 的格式组织
            normalize: 是否对向量进行L2归一化（默认True，推荐用于cosine相似度）
        
        Returns:
            向量数组 (n_samples, n_features)，已L2归一化（如果normalize=True）
        """
        # 如果提供了instruction，格式化文本
        if instruction:
            formatted_texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
        else:
            formatted_texts = texts
        
        if self.use_ollama:
            workers = batch_size if batch_size is not None else self.max_workers
            vectors = self._encode_ollama(formatted_texts, max_workers=workers)
        else:
            if self.model is None:
                self.load_model()
            bs = batch_size if batch_size is not None else self.batch_size
            vectors = self.model.encode(formatted_texts, batch_size=bs, show_progress_bar=False)
        
        # MRL维度裁剪（如果指定了维度且向量维度大于指定维度）
        if self.mrl_dimensions is not None and vectors.shape[1] > self.mrl_dimensions:
            # 直接截取前mrl_dimensions维（MRL模型已经训练好，前N维包含主要信息）
            vectors = vectors[:, :self.mrl_dimensions]
            logger.debug(f"MRL维度裁剪: {vectors.shape[1]} -> {self.mrl_dimensions}")
        
        # L2归一化（推荐用于cosine相似度）
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # 避免除零
            vectors = vectors / norms
        
        return vectors
    
    def _encode_ollama_single(self, text: str, index: int, total: int) -> tuple[int, np.ndarray]:
        """
        使用Ollama API编码单个文本（用于并发处理）
        
        Args:
            text: 文本内容
            index: 文本索引
            total: 总文本数
        
        Returns:
            (index, embedding_vector)
        """
        try:
            request_json = {
                "model": self.model_name,
                "prompt": text
            }
            # 如果指定了MRL维度，添加到请求中
            if self.mrl_dimensions is not None:
                request_json["options"] = {"dimensions": self.mrl_dimensions}
            
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=request_json,
                timeout=60
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            if (index + 1) % 10 == 0 or index == 0:
                logger.info(f"Embedding进度: {index + 1}/{total} ({(index + 1) * 100 // total}%)")
            return (index, np.array(embedding))
        except Exception as e:
            logger.error(f"Ollama embedding failed for text[{index}]: {text[:50]}... Error: {str(e)}")
            raise RuntimeError(f"Ollama embedding failed for text[{index}]: {text[:50]}... Error: {str(e)}")
    
    def _encode_ollama(self, texts: List[str], max_workers: int = 4) -> np.ndarray:
        """
        使用Ollama API进行embedding（并发版本）
        
        Args:
            texts: 文本列表
            max_workers: 最大并发数
        
        Returns:
            向量数组
        """
        if requests is None:
            raise ImportError("requests is required for Ollama embedding")
        
        total = len(texts)
        if total == 0:
            return np.array([])
        
        logger.info(f"开始Ollama embedding，共 {total} 个文本，并发数: {max_workers}")
        
        # 使用线程池并发处理
        vectors = [None] * total
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self._encode_ollama_single, text, idx, total): idx
                for idx, text in enumerate(texts)
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_index):
                try:
                    index, embedding = future.result()
                    vectors[index] = embedding
                    completed += 1
                except Exception as e:
                    idx = future_to_index[future]
                    logger.error(f"处理文本[{idx}]时出错: {str(e)}")
                    raise
        
        logger.info(f"Ollama embedding完成，共处理 {completed}/{total} 个文本")
        return np.array(vectors)
    
    def encode_single(
        self, 
        text: str, 
        instruction: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        编码单个文本
        
        Args:
            text: 文本内容
            instruction: 可选的instruction文本
            normalize: 是否L2归一化
        
        Returns:
            向量
        """
        return self.encode([text], instruction=instruction, normalize=normalize)[0]

