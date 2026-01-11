# 新增或修改我时需要修改这个文件夹中的README.md文件
"""Reranker工具 - 用于阶段D的二次验证"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import requests
except ImportError:
    requests = None

try:
    from ..models.llm_wrapper import LLMWrapper
except ImportError:
    LLMWrapper = None


class RerankerTool:
    """
    Reranker工具 - 用于阶段D的二次验证
    
    职责：
    - 对embedding近邻的候选对进行reranker打分
    - 支持Ollama reranker模型和LLM-based reranker
    - 用于边界精修，减少误聚/漏聚
    
    技术规范（基于docs/聚类规范.md）：
    - 输入格式：Instruct + Query + Document
    - 输出：P(yes)概率分数（0-1）
    - 只处理embedding近邻的候选对，成本可控
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        llm_wrapper: Optional[Any] = None,
        use_llm_reranker: bool = False,
        instruction: str = "Decide whether Query and Document describe the same customer issue type for clustering. Answer only yes or no.",
        score_threshold: float = 0.6,
        auto_fallback: bool = True  # 是否在Ollama失败时自动回退到LLM
    ):
        """
        Args:
            model_name: Reranker模型名称（Ollama模型，如 "qwen3-reranker:0.6b"）
            base_url: Ollama API地址（默认http://localhost:11434）
            llm_wrapper: LLM包装器（用于LLM-based reranker）
            use_llm_reranker: 是否使用LLM-based reranker（默认False，使用专用reranker模型）
            instruction: Reranker任务指令（默认英文，符合训练分布）
            score_threshold: 相关性分数阈值（默认0.6，低于此值的边将被丢弃）
            auto_fallback: 是否在Ollama reranker失败时自动回退到LLM-based reranker（默认True）
        """
        self.model_name = model_name
        self.base_url = base_url or "http://localhost:11434"
        self.llm_wrapper = llm_wrapper
        self.use_llm_reranker = use_llm_reranker
        self.instruction = instruction
        self.score_threshold = score_threshold
        self.max_workers = 4  # 默认并发数
        self.auto_fallback = auto_fallback
        self._ollama_available = None  # 缓存Ollama可用性检测结果
        self._fallback_triggered = False  # 是否已触发回退
        
        if not use_llm_reranker and not model_name:
            logger.warning("未指定reranker模型，将使用LLM-based reranker")
            self.use_llm_reranker = True
    
    def _check_ollama_available(self) -> bool:
        """
        检查Ollama reranker API是否可用
        
        Returns:
            True if available, False otherwise
        """
        if self._ollama_available is not None:
            return self._ollama_available
        
        if requests is None:
            self._ollama_available = False
            return False
        
        if not self.model_name:
            self._ollama_available = False
            return False
        
        try:
            # 尝试一个简单的健康检查（检查Ollama服务是否运行）
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # 检查rerank端点是否存在（通过OPTIONS请求或尝试一个测试请求）
                # 注意：Ollama可能不支持/api/rerank端点，这里只检查服务是否运行
                self._ollama_available = True
                return True
            else:
                self._ollama_available = False
                return False
        except Exception as e:
            logger.debug(f"Ollama服务检查失败: {str(e)}")
            self._ollama_available = False
            return False
    
    def _rerank_ollama_single(
        self, 
        query_text: str, 
        document_text: str,
        index: int,
        total: int
    ) -> Tuple[int, float]:
        """
        使用Ollama Reranker API对单个候选对打分
        
        Args:
            query_text: Query文本（样本i的cluster_text）
            document_text: Document文本（样本j的cluster_text）
            index: 候选对索引
            total: 总候选对数
        
        Returns:
            (index, score) - score是P(yes)概率（0-1）
        """
        if requests is None:
            raise ImportError("requests is required for Ollama reranker")
        
        try:
            # 构建reranker输入（按照Qwen3-Reranker的格式）
            # 格式：Instruct: {instruction}\nQuery: {query}\nDocument: {document}
            prompt = f"Instruct: {self.instruction}\nQuery: {query_text}\nDocument: {document_text}"
            
            response = requests.post(
                f"{self.base_url}/api/rerank",
                json={
                    "model": self.model_name,
                    "query": query_text,
                    "documents": [document_text],
                    "top_n": 1
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # 提取分数（reranker返回的是相关性分数，通常在0-1之间）
            if "results" in result and len(result["results"]) > 0:
                score = result["results"][0].get("relevance_score", 0.0)
            else:
                # 如果API格式不同，尝试其他字段
                score = result.get("score", 0.0)
            
            if (index + 1) % 50 == 0 or index == 0:
                logger.debug(f"Reranker进度: {index + 1}/{total} ({(index + 1) * 100 // total}%)")
            
            return (index, float(score))
        except requests.exceptions.HTTPError as e:
            # 404错误表示API端点不存在
            if e.response.status_code == 404:
                if not self._fallback_triggered and self.auto_fallback and self.llm_wrapper is not None:
                    logger.warning(
                        f"Ollama reranker API端点不存在 (404)，自动回退到LLM-based reranker。"
                        f"如果不想使用LLM reranker，请在配置中设置 use_reranker: false 或提供正确的reranker模型。"
                    )
                    self._fallback_triggered = True
                    self.use_llm_reranker = True
                    # 递归调用LLM reranker
                    return self._rerank_llm_single(query_text, document_text, index, total)
                else:
                    logger.error(
                        f"Ollama reranker API端点不存在 (404)。"
                        f"请检查：1) Ollama服务是否运行 2) 是否安装了reranker模型 3) Ollama版本是否支持rerank API。"
                        f"或者设置 use_reranker: false 来禁用reranker功能。"
                    )
            else:
                logger.error(f"Ollama reranker failed for pair[{index}]: HTTP {e.response.status_code} - {str(e)}")
            # 返回默认低分，避免阻塞
            return (index, 0.0)
        except Exception as e:
            logger.error(f"Ollama reranker failed for pair[{index}]: Error: {str(e)}")
            # 如果是第一次失败且启用了自动回退，尝试回退到LLM
            if not self._fallback_triggered and self.auto_fallback and self.llm_wrapper is not None:
                logger.warning(f"Ollama reranker失败，自动回退到LLM-based reranker")
                self._fallback_triggered = True
                self.use_llm_reranker = True
                return self._rerank_llm_single(query_text, document_text, index, total)
            # 返回默认低分，避免阻塞
            return (index, 0.0)
    
    def _rerank_llm_single(
        self,
        query_text: str,
        document_text: str,
        index: int,
        total: int
    ) -> Tuple[int, float]:
        """
        使用LLM-based reranker对单个候选对打分
        
        Args:
            query_text: Query文本
            document_text: Document文本
            index: 候选对索引
            total: 总候选对数
        
        Returns:
            (index, score) - score是P(yes)概率（0-1）
        """
        if self.llm_wrapper is None:
            raise ValueError("LLM wrapper is required for LLM-based reranker")
        
        try:
            # 构建prompt（二分类任务）
            prompt = f"""{self.instruction}

Query: {query_text}
Document: {document_text}

Please answer only "yes" or "no"."""
            
            # 调用LLM
            response = self.llm_wrapper.llm.invoke(prompt)
            response_text = response.content.strip().lower()
            
            # 解析yes/no（简单实现，可以改进为概率输出）
            if "yes" in response_text:
                score = 0.8  # 如果回答yes，给较高分数
            elif "no" in response_text:
                score = 0.2  # 如果回答no，给较低分数
            else:
                # 无法确定，给中等分数
                score = 0.5
            
            if (index + 1) % 50 == 0 or index == 0:
                logger.debug(f"LLM Reranker进度: {index + 1}/{total} ({(index + 1) * 100 // total}%)")
            
            return (index, float(score))
        except Exception as e:
            logger.error(f"LLM reranker failed for pair[{index}]: Error: {str(e)}")
            return (index, 0.0)
    
    def rerank_pairs(
        self,
        query_texts: List[str],
        document_texts: List[str],
        max_workers: int = 4
    ) -> np.ndarray:
        """
        对候选对列表进行reranker打分
        
        Args:
            query_texts: Query文本列表
            document_texts: Document文本列表（与query_texts一一对应）
            max_workers: 并发数（默认4）
        
        Returns:
            分数数组（与输入一一对应）
        """
        if len(query_texts) != len(document_texts):
            raise ValueError("query_texts and document_texts must have the same length")
        
        total = len(query_texts)
        if total == 0:
            return np.array([])
        
        # 检查是否需要回退到LLM reranker
        if not self.use_llm_reranker and self.auto_fallback:
            if not self._check_ollama_available():
                if self.llm_wrapper is not None:
                    logger.warning("Ollama reranker不可用，自动切换到LLM-based reranker")
                    self.use_llm_reranker = True
                else:
                    logger.error("Ollama reranker不可用且未提供LLM wrapper，无法回退。请检查配置或禁用reranker。")
        
        logger.info(f"开始Reranker打分，共 {total} 个候选对，并发数: {max_workers}，模式: {'LLM-based' if self.use_llm_reranker else 'Ollama'}")
        
        scores = [0.0] * total
        
        if self.use_llm_reranker:
            # LLM-based reranker（串行处理，避免过载）
            if self.llm_wrapper is None:
                logger.error("LLM wrapper未提供，无法使用LLM-based reranker")
                return np.array(scores)
            for idx, (query, doc) in enumerate(zip(query_texts, document_texts)):
                _, score = self._rerank_llm_single(query, doc, idx, total)
                scores[idx] = score
        else:
            # Ollama reranker（并发处理）
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        self._rerank_ollama_single, 
                        query, 
                        doc, 
                        idx, 
                        total
                    ): idx
                    for idx, (query, doc) in enumerate(zip(query_texts, document_texts))
                }
                
                completed = 0
                for future in as_completed(future_to_index):
                    try:
                        index, score = future.result()
                        scores[index] = score
                        completed += 1
                    except Exception as e:
                        idx = future_to_index[future]
                        logger.error(f"处理候选对[{idx}]时出错: {str(e)}")
                        scores[idx] = 0.0
        
        logger.info(f"Reranker打分完成，共处理 {len(scores)} 个候选对")
        return np.array(scores)
    
    def filter_by_score(
        self,
        scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        根据分数阈值过滤候选对
        
        Args:
            scores: 分数数组
            threshold: 阈值（默认使用self.score_threshold）
        
        Returns:
            布尔数组，True表示保留，False表示丢弃
        """
        threshold = threshold or self.score_threshold
        return scores >= threshold

