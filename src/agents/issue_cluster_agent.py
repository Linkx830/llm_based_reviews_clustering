# 新增或修改我时需要修改这个文件夹中的README.md文件
"""IssueClusterAgent - 两阶段聚类：aspect分桶 + 桶内issue聚类"""
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict
import json
import numpy as np
from .base_agent import BaseAgent
from ..storage.table_manager import TableManager
from ..tools.embedding_tool import EmbeddingTool
from ..tools.clustering_tool import ClusteringTool
from ..tools.reranker_tool import RerankerTool
from ..utils.logger import get_logger

logger = get_logger(__name__)


class IssueClusterAgent(BaseAgent):
    """
    IssueClusterAgent - 基于新聚类规范的两阶段聚类
    
    职责：
    - 对VALID记录进行两阶段聚类
    - 阶段1：Aspect分桶（合并同义aspect）
    - 阶段2：桶内对Issue聚类
    - 使用结构化输入 + instruction-aware embedding
    - 输出簇归属与簇统计
    
    聚类策略（基于docs/聚类规范.md）：
    - 结构化输入：Aspect: {aspect_norm}\nIssue: {issue_norm}
    - 两路向量：E_issue（主向量）和 E_aspect（辅向量）
    - 两阶段聚类：先按aspect分桶，再桶内聚类issue
    - 簇后处理：medoid选择、噪声处理
    
    输入表：aspect_sentiment_valid
    输出表：issue_clusters, cluster_stats
    """
    
    def __init__(
        self,
        *args,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_base_url: str = None,
        embedding_max_workers: int = 4,
        embedding_batch_size: int = 32,
        embedding_mrl_dimensions: Optional[int] = None,  # MRL维度裁剪
        clustering_config: Dict[str, Any] = None,
        auto_select_threshold: int = 1000,
        # 新聚类规范参数
        use_instruction: bool = True,
        issue_instruction: str = "Represent the underlying customer issue type for clustering. Texts should be close if they describe the same issue type, even with different wording.",
        aspect_instruction: str = "Represent the product aspect category. Texts should be close if they refer to the same aspect.",
        aspect_similarity_threshold: float = 0.85,  # aspect同义合并的相似度阈值
        use_combined_vector: bool = False,  # 是否使用拼接向量（默认False，使用两阶段聚类）
        issue_weight: float = 0.8,  # 拼接向量时issue的权重
        aspect_weight: float = 0.2,  # 拼接向量时aspect的权重
        # 阶段D：Reranker二次验证参数
        use_reranker: bool = False,  # 是否启用reranker二次验证
        reranker_model: Optional[str] = None,  # Reranker模型名称
        reranker_base_url: Optional[str] = None,  # Reranker API地址
        reranker_llm_wrapper: Optional[Any] = None,  # LLM包装器（用于LLM-based reranker）
        reranker_top_k: int = 50,  # 每个样本的reranker候选数（默认50）
        reranker_score_threshold: float = 0.6,  # Reranker分数阈值
        reranker_max_workers: int = 4,  # Reranker并发数
        # 阶段E：噪点簇和小簇后处理参数
        min_cluster_size: int = 2,  # 最小簇大小（小于此值的簇将被处理）
        noise_adsorption_threshold: float = 0.7,  # 噪点吸附到最近簇的相似度阈值
        small_cluster_merge_threshold: float = 0.75,  # 小簇合并的相似度阈值
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model
        self.auto_select_threshold = auto_select_threshold
        self.clustering_config = clustering_config or {}
        self.use_instruction = use_instruction
        self.issue_instruction = issue_instruction
        self.aspect_instruction = aspect_instruction
        self.aspect_similarity_threshold = aspect_similarity_threshold
        self.use_combined_vector = use_combined_vector
        self.issue_weight = issue_weight
        self.aspect_weight = aspect_weight
        self.use_reranker = use_reranker
        self.reranker_top_k = reranker_top_k
        self.reranker_score_threshold = reranker_score_threshold
        self.min_cluster_size = min_cluster_size
        self.noise_adsorption_threshold = noise_adsorption_threshold
        self.small_cluster_merge_threshold = small_cluster_merge_threshold
        
        self.embedding_tool = EmbeddingTool(
            embedding_model, 
            base_url=embedding_base_url,
            max_workers=embedding_max_workers,
            batch_size=embedding_batch_size,
            mrl_dimensions=embedding_mrl_dimensions
        )
        
        # 初始化Reranker工具（如果启用）
        if self.use_reranker:
            reranker_tool = RerankerTool(
                model_name=reranker_model,
                base_url=reranker_base_url or embedding_base_url,
                llm_wrapper=reranker_llm_wrapper,
                use_llm_reranker=(reranker_llm_wrapper is not None),
                score_threshold=reranker_score_threshold
            )
            reranker_tool.max_workers = reranker_max_workers
            self.reranker_tool = reranker_tool
        else:
            self.reranker_tool = None
    
    def _build_cluster_text(self, aspect_norm: str, issue_norm: str) -> str:
        """
        构建结构化聚类文本（阶段A）
        
        Args:
            aspect_norm: 规范化后的aspect
            issue_norm: 规范化后的issue
        
        Returns:
            结构化文本：Aspect: {aspect_norm}\nIssue: {issue_norm}
        """
        return f"Aspect: {aspect_norm}\nIssue: {issue_norm}"
    
    def _merge_similar_aspects(
        self, 
        aspect_norms: List[str], 
        aspect_vectors: np.ndarray
    ) -> Dict[str, str]:
        """
        合并同义aspect（阶段C1）
        
        Args:
            aspect_norms: aspect列表
            aspect_vectors: aspect向量数组
        
        Returns:
            aspect到标准aspect的映射字典
        """
        logger.info(f"开始合并同义aspect，共 {len(aspect_norms)} 个aspect")
        
        # 使用cosine相似度进行聚类
        from sklearn.cluster import AgglomerativeClustering
        
        if len(aspect_norms) <= 1:
            return {aspect: aspect for aspect in aspect_norms}
        
        # 计算cosine距离矩阵（1 - cosine相似度）
        # 由于向量已归一化，cosine相似度 = dot product
        similarity_matrix = np.dot(aspect_vectors, aspect_vectors.T)
        distance_matrix = 1 - similarity_matrix
        
        # 使用AgglomerativeClustering合并相似aspect
        # 距离阈值 = 1 - 相似度阈值
        distance_threshold = 1 - self.aspect_similarity_threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="average",
            metric="precomputed"
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # 为每个簇选择标准aspect（选择频次最高的）
        aspect_to_canonical = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_aspects = [aspect_norms[i] for i in range(len(aspect_norms)) if labels[i] == label]
            if cluster_aspects:
                # 选择频次最高的aspect作为标准aspect
                aspect_counter = Counter(cluster_aspects)
                canonical_aspect = aspect_counter.most_common(1)[0][0]
                for aspect in cluster_aspects:
                    aspect_to_canonical[aspect] = canonical_aspect
        
        merged_count = len(unique_labels)
        logger.info(
            f"Aspect合并完成: {len(aspect_norms)} 个aspect合并为 {merged_count} 个标准aspect "
            f"(合并率: {(1 - merged_count / len(aspect_norms)) * 100:.1f}%)"
        )
        
        return aspect_to_canonical
    
    def _select_medoid(
        self, 
        vectors: np.ndarray, 
        labels: np.ndarray, 
        cluster_id: int
    ) -> int:
        """
        选择簇的medoid（代表样本）
        
        Args:
            vectors: 向量数组
            labels: 簇标签
            cluster_id: 目标簇ID
        
        Returns:
            medoid的索引
        """
        mask = labels == cluster_id
        cluster_vectors = vectors[mask]
        cluster_indices = np.where(mask)[0]
        
        if len(cluster_vectors) == 1:
            return cluster_indices[0]
        
        # 计算簇内每条样本到其他样本的平均相似度（cosine）
        # 由于向量已归一化，cosine相似度 = dot product
        similarities = np.dot(cluster_vectors, cluster_vectors.T)
        # 排除自身（对角线）
        np.fill_diagonal(similarities, 0)
        # 计算平均相似度
        avg_similarities = np.mean(similarities, axis=1)
        # 选择平均相似度最高的作为medoid
        medoid_idx_in_cluster = np.argmax(avg_similarities)
        return cluster_indices[medoid_idx_in_cluster]
    
    def _reranker_refinement(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        cluster_texts: List[str],
        bucket_vectors: np.ndarray,
        bucket_indices: List[int]
    ) -> np.ndarray:
        """
        阶段D：使用Reranker进行边界精修
        
        Args:
            vectors: 所有向量（用于kNN搜索）
            labels: 当前簇标签
            cluster_texts: 所有cluster_text列表
            bucket_vectors: 当前桶的向量
            bucket_indices: 当前桶的全局索引
        
        Returns:
            精修后的标签数组（仅针对当前桶）
        """
        if not self.use_reranker or self.reranker_tool is None:
            return labels
        
        logger.info(f"阶段D: 开始Reranker边界精修，桶大小: {len(bucket_vectors)}")
        
        # 对每个样本，找到top-k近邻
        from sklearn.neighbors import NearestNeighbors
        
        # 使用cosine距离（向量已归一化）
        nbrs = NearestNeighbors(n_neighbors=min(self.reranker_top_k + 1, len(vectors)), metric='cosine')
        nbrs.fit(vectors)
        
        # 为每个桶内样本找到近邻
        query_texts = []
        document_texts = []
        pair_to_bucket_idx = []  # 记录每个候选对对应的桶内索引
        
        for bucket_idx, bucket_vec in enumerate(bucket_vectors):
            # 找到近邻（包括自身）
            distances, indices = nbrs.kneighbors([bucket_vec], n_neighbors=min(self.reranker_top_k + 1, len(vectors)))
            
            # 排除自身，取top-k
            neighbor_indices = indices[0][1:self.reranker_top_k + 1]
            
            query_text = cluster_texts[bucket_indices[bucket_idx]]
            for neighbor_idx in neighbor_indices:
                document_text = cluster_texts[neighbor_idx]
                query_texts.append(query_text)
                document_texts.append(document_text)
                pair_to_bucket_idx.append(bucket_idx)
        
        if not query_texts:
            return labels
        
        # Reranker打分
        logger.info(f"Reranker候选对数: {len(query_texts)}")
        scores = self.reranker_tool.rerank_pairs(
            query_texts,
            document_texts,
            max_workers=self.reranker_tool.max_workers
        )
        
        # 构建图：节点=桶内样本，边=reranker高分对
        from collections import defaultdict
        
        # 为每个桶内样本，保留reranker分数最高的top-m条边
        top_m = 10  # 每个样本最多保留10条边
        
        # 构建全局索引到桶内索引的映射
        global_to_bucket = {global_idx: bucket_idx for bucket_idx, global_idx in enumerate(bucket_indices)}
        
        # 重建neighbor_indices列表（因为之前是循环生成的）
        neighbor_idx_list = []
        for bucket_idx in range(len(bucket_vectors)):
            distances, indices = nbrs.kneighbors([bucket_vectors[bucket_idx]], n_neighbors=min(self.reranker_top_k + 1, len(vectors)))
            neighbor_idx_list.extend(indices[0][1:self.reranker_top_k + 1])
        
        # 构建边字典：{bucket_idx: [(neighbor_bucket_idx, score), ...]}
        # 只保留邻居也在桶内的边
        edges = defaultdict(list)
        for pair_idx, bucket_idx in enumerate(pair_to_bucket_idx):
            score = scores[pair_idx]
            neighbor_global_idx = neighbor_idx_list[pair_idx]
            
            if score >= self.reranker_score_threshold:
                # 检查neighbor是否也在桶内
                if neighbor_global_idx in global_to_bucket:
                    neighbor_bucket_idx = global_to_bucket[neighbor_global_idx]
                    # 避免自环
                    if neighbor_bucket_idx != bucket_idx:
                        edges[bucket_idx].append((neighbor_bucket_idx, score))
        
        # 对每个桶内样本，只保留top-m条边
        for bucket_idx in edges:
            edges[bucket_idx].sort(key=lambda x: x[1], reverse=True)
            edges[bucket_idx] = edges[bucket_idx][:top_m]
        
        # 使用reranker分数重新聚类（图聚类）
        logger.info("使用Reranker分数进行图精修聚类...")
        
        # 构建无向图（双向边，取平均分数）
        graph = defaultdict(dict)  # {node: {neighbor: max_score}}
        for bucket_idx, neighbor_list in edges.items():
            for neighbor_bucket_idx, score in neighbor_list:
                # 构建双向边，取最大分数（或平均分数）
                if neighbor_bucket_idx not in graph[bucket_idx]:
                    graph[bucket_idx][neighbor_bucket_idx] = score
                else:
                    graph[bucket_idx][neighbor_bucket_idx] = max(
                        graph[bucket_idx][neighbor_bucket_idx], score
                    )
                # 反向边
                if bucket_idx not in graph[neighbor_bucket_idx]:
                    graph[neighbor_bucket_idx][bucket_idx] = score
                else:
                    graph[neighbor_bucket_idx][bucket_idx] = max(
                        graph[neighbor_bucket_idx][bucket_idx], score
                    )
        
        # 使用连通分量进行图聚类
        def find_connected_components(graph, n_nodes):
            """
            使用DFS找到所有连通分量
            
            Args:
                graph: 邻接表表示的图 {node: {neighbor: score}}
                n_nodes: 节点总数
            
            Returns:
                标签数组，每个节点属于哪个连通分量
            """
            visited = [False] * n_nodes
            labels = np.full(n_nodes, -1, dtype=int)
            component_id = 0
            
            def dfs(node, comp_id):
                """深度优先搜索"""
                stack = [node]
                visited[node] = True
                labels[node] = comp_id
                
                while stack:
                    current = stack.pop()
                    for neighbor in graph.get(current, {}).keys():
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            labels[neighbor] = comp_id
                            stack.append(neighbor)
            
            for node in range(n_nodes):
                if not visited[node]:
                    dfs(node, component_id)
                    component_id += 1
            
            return labels, component_id
        
        # 执行连通分量聚类
        reranker_labels, num_components = find_connected_components(graph, len(bucket_vectors))
        
        logger.info(
            f"Reranker图聚类完成: 发现 {num_components} 个连通分量 "
            f"(原簇数: {len(set(labels)) - (1 if -1 in labels else 0)})"
        )
        
        # 基于reranker图聚类结果重新分配标签
        # 策略：reranker的高置信度连接优先于原标签
        # 1. 如果reranker确认两个样本应该在同一簇（在同一个连通分量），强制它们在同一簇
        # 2. 对于噪声点，如果reranker确认它应该属于某个簇，则将其加入该簇
        # 3. 对于reranker图中孤立的节点，保持原标签
        
        refined_labels = labels.copy()
        
        # 为每个reranker连通分量，找到其包含的原标签和节点
        component_to_nodes = defaultdict(list)  # {comp_id: [bucket_idx, ...]}
        component_to_original_labels = defaultdict(set)  # {comp_id: {label1, label2, ...}}
        
        for bucket_idx, reranker_label in enumerate(reranker_labels):
            component_to_nodes[reranker_label].append(bucket_idx)
            original_label = labels[bucket_idx]
            if original_label != -1:  # 忽略噪声点
                component_to_original_labels[reranker_label].add(original_label)
        
        # 对于每个reranker连通分量，决定如何重新分配标签
        for comp_id, nodes in component_to_nodes.items():
            original_label_set = component_to_original_labels[comp_id]
            
            if len(original_label_set) == 0:
                # 该连通分量只包含噪声点，保持为噪声
                continue
            elif len(original_label_set) == 1:
                # 该连通分量只包含一个原标签，保持原标签（但将噪声点加入该簇）
                target_label = list(original_label_set)[0]
                for bucket_idx in nodes:
                    if labels[bucket_idx] == -1:
                        # 噪声点被reranker确认应该属于该簇
                        refined_labels[bucket_idx] = target_label
            else:
                # 该连通分量包含多个原标签，需要合并
                # 使用最小的标签ID作为目标标签
                target_label = min(original_label_set)
                for bucket_idx in nodes:
                    refined_labels[bucket_idx] = target_label
        
        # 对于reranker图中孤立的节点（没有边的节点），保持原标签
        # 孤立节点：不在graph中，也不在任何节点的neighbors中
        all_nodes_in_graph = set(graph.keys())
        for node, neighbors in graph.items():
            all_nodes_in_graph.update(neighbors.keys())
        isolated_nodes = set(range(len(bucket_vectors))) - all_nodes_in_graph
        
        for node in isolated_nodes:
            # 保持原标签（reranker没有提供足够的信息来改变它）
            refined_labels[node] = labels[node]
        
        # 重新编号标签（确保连续）
        unique_refined_labels = sorted(set(refined_labels))
        if -1 in unique_refined_labels:
            unique_refined_labels.remove(-1)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_refined_labels)}
        label_mapping[-1] = -1  # 保持噪声标签
        
        final_labels = np.array([label_mapping[label] for label in refined_labels])
        
        # 统计变化
        changed_count = sum(1 for i in range(len(labels)) if labels[i] != final_labels[i])
        logger.info(
            f"Reranker精修完成: {changed_count}/{len(labels)} 个样本的标签发生变化 "
            f"(变化率: {changed_count/len(labels)*100:.1f}%)"
        )
        
        return final_labels
    
    def _post_process_clusters(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        cluster_texts: List[str],
        all_sentence_ids: List[str],
        all_aspect_norms: List[str]
    ) -> np.ndarray:
        """
        阶段E：噪点簇和小簇后处理
        
        Args:
            vectors: 向量数组
            labels: 簇标签
            cluster_texts: cluster_text列表
            all_sentence_ids: 句子ID列表
            all_aspect_norms: aspect列表
        
        Returns:
            处理后的标签数组
        """
        logger.info("阶段E: 开始噪点簇和小簇后处理")
        
        unique_labels = np.unique(labels)
        noise_labels = labels == -1
        noise_count = noise_labels.sum()
        
        logger.info(f"处理前: 总簇数={len(unique_labels) - (1 if -1 in unique_labels else 0)}, 噪声点={noise_count}")
        
        # 1. 噪点吸附：将噪声点吸附到最近的簇
        if noise_count > 0:
            logger.info(f"开始噪点吸附，噪声点数: {noise_count}")
            noise_indices = np.where(noise_labels)[0]
            valid_cluster_labels = [l for l in unique_labels if l != -1]
            
            if valid_cluster_labels:
                # 计算每个噪声点到各簇medoid的距离
                from sklearn.neighbors import NearestNeighbors
                
                # 找到每个簇的medoid
                cluster_medoids = {}
                for cluster_id in valid_cluster_labels:
                    medoid_idx = self._select_medoid(vectors, labels, cluster_id)
                    cluster_medoids[cluster_id] = medoid_idx
                
                # 对噪声点，找到最近的簇medoid
                noise_vectors = vectors[noise_indices]
                medoid_vectors = np.array([vectors[cluster_medoids[cid]] for cid in valid_cluster_labels])
                
                # 计算cosine相似度（向量已归一化）
                similarities = np.dot(noise_vectors, medoid_vectors.T)
                
                # 为每个噪声点找到最相似的簇
                for i, noise_idx in enumerate(noise_indices):
                    max_sim_idx = np.argmax(similarities[i])
                    max_similarity = similarities[i][max_sim_idx]
                    
                    if max_similarity >= self.noise_adsorption_threshold:
                        target_cluster = valid_cluster_labels[max_sim_idx]
                        labels[noise_idx] = target_cluster
                        logger.debug(
                            f"噪声点 {noise_idx} 吸附到簇 {target_cluster} "
                            f"(相似度: {max_similarity:.3f})"
                        )
                
                adsorbed_count = (labels[noise_indices] != -1).sum()
                logger.info(f"噪点吸附完成: {adsorbed_count}/{noise_count} 个噪声点被吸附")
        
        # 2. 小簇处理：合并小簇或标记为噪声
        logger.info("开始小簇处理...")
        unique_labels = np.unique(labels)
        valid_cluster_labels = [l for l in unique_labels if l != -1]
        
        small_clusters = []
        for cluster_id in valid_cluster_labels:
            cluster_size = (labels == cluster_id).sum()
            if cluster_size < self.min_cluster_size:
                small_clusters.append(cluster_id)
        
        if small_clusters:
            logger.info(f"发现 {len(small_clusters)} 个小簇（大小 < {self.min_cluster_size}）")
            
            # 计算小簇的medoid
            small_cluster_medoids = {}
            for cluster_id in small_clusters:
                medoid_idx = self._select_medoid(vectors, labels, cluster_id)
                small_cluster_medoids[cluster_id] = medoid_idx
            
            # 尝试将小簇合并到相似的大簇
            large_cluster_labels = [l for l in valid_cluster_labels if l not in small_clusters]
            
            if large_cluster_labels:
                # 计算大簇的medoid
                large_cluster_medoids = {}
                for cluster_id in large_cluster_labels:
                    medoid_idx = self._select_medoid(vectors, labels, cluster_id)
                    large_cluster_medoids[cluster_id] = medoid_idx
                
                large_medoid_vectors = np.array([vectors[large_cluster_medoids[cid]] for cid in large_cluster_labels])
                
                merged_count = 0
                for small_cluster_id in small_clusters:
                    small_medoid_idx = small_cluster_medoids[small_cluster_id]
                    small_medoid_vec = vectors[small_medoid_idx:small_medoid_idx+1]
                    
                    # 计算到各大簇medoid的相似度
                    similarities = np.dot(small_medoid_vec, large_medoid_vectors.T)[0]
                    max_sim_idx = np.argmax(similarities)
                    max_similarity = similarities[max_sim_idx]
                    
                    if max_similarity >= self.small_cluster_merge_threshold:
                        target_cluster = large_cluster_labels[max_sim_idx]
                        # 合并小簇到大簇
                        labels[labels == small_cluster_id] = target_cluster
                        merged_count += 1
                        logger.debug(
                            f"小簇 {small_cluster_id} 合并到簇 {target_cluster} "
                            f"(相似度: {max_similarity:.3f})"
                        )
                    else:
                        # 相似度不够，标记为噪声
                        labels[labels == small_cluster_id] = -1
                        logger.debug(
                            f"小簇 {small_cluster_id} 相似度不足，标记为噪声 "
                            f"(最大相似度: {max_similarity:.3f})"
                        )
                
                logger.info(f"小簇处理完成: {merged_count} 个合并，{len(small_clusters) - merged_count} 个标记为噪声")
            else:
                # 没有大簇，所有小簇标记为噪声
                for small_cluster_id in small_clusters:
                    labels[labels == small_cluster_id] = -1
                logger.info(f"没有大簇，所有 {len(small_clusters)} 个小簇标记为噪声")
        
        final_unique_labels = np.unique(labels)
        final_noise_count = (labels == -1).sum()
        final_cluster_count = len(final_unique_labels) - (1 if -1 in final_unique_labels else 0)
        
        logger.info(
            f"后处理完成: 总簇数={final_cluster_count}, 噪声点={final_noise_count}"
        )
        
        return labels
    
    def process(self) -> Dict[str, Any]:
        """执行两阶段聚类"""
        logger.info(
            f"开始执行IssueClusterAgent（两阶段聚类），run_id={self.run_id}, "
            f"embedding_model={self.embedding_model}"
        )
        
        # 读取VALID记录
        query = f"""
            SELECT sentence_id, aspect_norm, issue_norm, sentiment
            FROM {TableManager.ASPECT_SENTIMENT_VALID}
            WHERE run_id = ? AND validity_label = 'VALID'
        """
        valid_records = self.db.execute_read(query, [self.run_id])
        logger.info(f"读取到 {len(valid_records)} 条VALID记录")
        
        if not valid_records:
            logger.warning("没有VALID记录，跳过聚类")
            return {
                "status": "success",
                "cluster_count": 0,
                "table": TableManager.ISSUE_CLUSTERS
            }
        
        # 解析记录
        sentence_ids = []
        aspect_norms = []
        issue_norms = []
        sentiments = []
        
        for row in valid_records:
            sentence_id, aspect_norm, issue_norm, sentiment = row
            sentence_ids.append(sentence_id)
            aspect_norms.append(aspect_norm)
            issue_norms.append(issue_norm)
            sentiments.append(sentiment)
        
        total_count = len(sentence_ids)
        
        # ========== 阶段A：构建结构化输入 ==========
        logger.info("阶段A: 构建结构化输入文本")
        cluster_texts = []
        for aspect_norm, issue_norm in zip(aspect_norms, issue_norms):
            cluster_text = self._build_cluster_text(aspect_norm, issue_norm)
            cluster_texts.append(cluster_text)
        
        # ========== 阶段B：生成两路向量 ==========
        logger.info("阶段B: 生成两路向量（E_issue 和 E_aspect）")
        
        # 去重：只对唯一文本进行embedding
        unique_cluster_texts = list(set(cluster_texts))
        unique_aspects = list(set(aspect_norms))
        
        logger.info(
            f"唯一cluster_text: {len(unique_cluster_texts)}, "
            f"唯一aspect: {len(unique_aspects)}"
        )
        
        # 生成E_issue向量（主向量）
        issue_instruction = self.issue_instruction if self.use_instruction else None
        unique_issue_vectors = self.embedding_tool.encode(
            unique_cluster_texts,
            instruction=issue_instruction,
            normalize=True
        )
        logger.info(f"E_issue向量生成完成，维度: {unique_issue_vectors.shape[1]}")
        
        # 生成E_aspect向量（辅向量）
        aspect_instruction = self.aspect_instruction if self.use_instruction else None
        unique_aspect_vectors = self.embedding_tool.encode(
            unique_aspects,
            instruction=aspect_instruction,
            normalize=True
        )
        logger.info(f"E_aspect向量生成完成，维度: {unique_aspect_vectors.shape[1]}")
        
        # 构建文本到向量的映射
        cluster_text_to_issue_vec = {
            text: vec for text, vec in zip(unique_cluster_texts, unique_issue_vectors)
        }
        aspect_to_aspect_vec = {
            aspect: vec for aspect, vec in zip(unique_aspects, unique_aspect_vectors)
        }
        
        # 为所有记录分配向量
        issue_vectors = np.array([
            cluster_text_to_issue_vec[cluster_text] 
            for cluster_text in cluster_texts
        ])
        aspect_vectors = np.array([
            aspect_to_aspect_vec[aspect_norm]
            for aspect_norm in aspect_norms
        ])
        
        # ========== 阶段C1：Aspect分桶（合并同义aspect） ==========
        logger.info("阶段C1: Aspect分桶（合并同义aspect）")
        aspect_to_canonical = self._merge_similar_aspects(
            unique_aspects, 
            unique_aspect_vectors
        )
        
        # 应用合并映射
        canonical_aspects = [aspect_to_canonical.get(asp, asp) for asp in aspect_norms]
        
        # 按标准aspect分组
        aspect_buckets = defaultdict(list)
        for idx, canonical_aspect in enumerate(canonical_aspects):
            aspect_buckets[canonical_aspect].append({
                'index': idx,
                'sentence_id': sentence_ids[idx],
                'aspect_norm': canonical_aspect,
                'original_aspect': aspect_norms[idx],
                'issue_norm': issue_norms[idx],
                'sentiment': sentiments[idx],
                'issue_vector': issue_vectors[idx]
            })
        
        logger.info(f"按aspect分桶: {len(aspect_buckets)} 个aspect桶")
        for aspect, records in aspect_buckets.items():
            logger.debug(f"  Aspect '{aspect}': {len(records)} 条记录")
        
        # ========== 阶段C2：桶内对Issue聚类 ==========
        logger.info("阶段C2: 桶内对Issue聚类")
        
        all_labels = []
        all_sentence_ids = []
        all_aspect_norms = []
        all_issue_norms = []
        all_sentiments = []
        all_vectors = []
        global_cluster_id = 0  # 全局簇ID（跨aspect唯一）
        
        for canonical_aspect, records in aspect_buckets.items():
            bucket_size = len(records)
            
            if bucket_size < 2:
                # 单个记录无法聚类，直接标记为簇
                for record in records:
                    all_labels.append(global_cluster_id)
                    all_sentence_ids.append(record['sentence_id'])
                    all_aspect_norms.append(record['aspect_norm'])
                    all_issue_norms.append(record['issue_norm'])
                    all_sentiments.append(record['sentiment'])
                    all_vectors.append(record['issue_vector'])
                global_cluster_id += 1
                logger.debug(f"Aspect '{canonical_aspect}': 单样本，跳过聚类")
                continue
            
            # 提取该桶的issue向量
            bucket_vectors = np.array([r['issue_vector'] for r in records])
            
            # 根据数据量选择聚类方法
            if bucket_size < self.auto_select_threshold:
                method = "agglomerative"
            else:
                method = "hdbscan"
            
            # 合并用户配置
            final_config = {
                "method": method,
                **self.clustering_config
            }
            if "method" in self.clustering_config:
                final_config["method"] = self.clustering_config["method"]
            
            # 自适应参数（根据规范建议）
            if method == "hdbscan":
                min_cluster_size = final_config.get("min_cluster_size")
                if min_cluster_size is None:
                    # 根据规范：max(10, n * 0.005)
                    final_config["min_cluster_size"] = max(10, int(bucket_size * 0.005))
                min_samples = final_config.get("min_samples", 3)
                final_config["min_samples"] = min_samples
            elif method == "agglomerative":
                # Agglomerative使用distance_threshold自动确定簇数
                if "distance_threshold" not in final_config:
                    final_config["distance_threshold"] = 0.5
            
            # 聚类
            clustering_tool = ClusteringTool(**final_config)
            # 使用cosine距离（向量已归一化，可以直接用）
            labels = clustering_tool.fit(bucket_vectors)
            
            # ========== 阶段D：Reranker边界精修（可选） ==========
            if self.use_reranker and bucket_size >= 3:  # 至少3个样本才值得reranker
                bucket_cluster_texts = [
                    self._build_cluster_text(record['aspect_norm'], record['issue_norm'])
                    for record in records
                ]
                # 注意：这里需要在全局向量空间中搜索，但当前只有桶内向量
                # 简化实现：在桶内进行reranker精修
                labels = self._reranker_refinement(
                    bucket_vectors,  # 使用桶内向量作为搜索空间
                    labels,
                    bucket_cluster_texts,
                    bucket_vectors,
                    list(range(len(records)))  # 桶内索引
                )
            
            # 将局部簇ID转换为全局簇ID
            unique_local_labels = set(labels)
            local_to_global = {}
            for local_label in unique_local_labels:
                if local_label == -1:  # 噪声点
                    local_to_global[-1] = -1
                else:
                    local_to_global[local_label] = global_cluster_id
                    global_cluster_id += 1
            
            # 转换标签并收集结果
            for i, local_label in enumerate(labels):
                global_label = local_to_global[local_label]
                all_labels.append(global_label)
                all_sentence_ids.append(records[i]['sentence_id'])
                all_aspect_norms.append(records[i]['aspect_norm'])
                all_issue_norms.append(records[i]['issue_norm'])
                all_sentiments.append(records[i]['sentiment'])
                all_vectors.append(bucket_vectors[i])
            
            noise_count = sum(1 for l in labels if l == -1)
            cluster_count = len(unique_local_labels) - (1 if -1 in unique_local_labels else 0)
            logger.info(
                f"Aspect '{canonical_aspect}': {bucket_size} 条记录, "
                f"聚类得到 {cluster_count} 个簇, {noise_count} 个噪声点"
            )
        
        # 转换为numpy数组
        all_vectors = np.array(all_vectors)
        all_labels = np.array(all_labels)
        
        total_clusters = len(set(all_labels)) - (1 if -1 in all_labels else 0)
        total_noise = sum(1 for l in all_labels if l == -1)
        logger.info(
            f"全局聚类完成: 总簇数={total_clusters}, 噪声点={total_noise}"
        )
        
        # ========== 阶段E：簇后处理（噪点簇和小簇处理） ==========
        logger.info("阶段E: 开始簇后处理（噪点簇和小簇处理）")
        all_cluster_texts = [
            self._build_cluster_text(aspect, issue)
            for aspect, issue in zip(all_aspect_norms, all_issue_norms)
        ]
        all_labels = self._post_process_clusters(
            all_vectors,
            all_labels,
            all_cluster_texts,
            all_sentence_ids,
            all_aspect_norms
        )
        
        # ========== 阶段E续：计算统计、选择medoid ==========
        logger.info("阶段E续: 计算簇统计、选择medoid")
        
        # 计算簇统计
        clustering_tool = ClusteringTool(**final_config)
        cluster_stats = clustering_tool.compute_cluster_stats(
            all_labels, all_vectors, all_sentiments, all_sentence_ids
        )
        
        # 为每个簇选择medoid和确定主要aspect
        for cluster_id in cluster_stats.keys():
            cluster_label = int(cluster_id)
            
            # 选择medoid
            medoid_idx = self._select_medoid(all_vectors, all_labels, cluster_label)
            cluster_stats[cluster_id]["medoid_sentence_id"] = all_sentence_ids[medoid_idx]
            
            # 确定主要aspect（应该已经是同一个aspect，因为按aspect分桶了）
            cluster_aspects = [
                all_aspect_norms[i] 
                for i in range(len(all_labels)) 
                if all_labels[i] == cluster_label
            ]
            if cluster_aspects:
                aspect_counter = Counter(cluster_aspects)
                main_aspect = aspect_counter.most_common(1)[0][0]
                cluster_stats[cluster_id]["aspect_norm"] = main_aspect
            else:
                cluster_stats[cluster_id]["aspect_norm"] = ""
        
        # ========== 插入数据库 ==========
        logger.info("开始插入数据库...")
        table_manager = TableManager(self.db)
        
        # 构建最终配置（包含新聚类规范的参数）
        final_config_with_meta = {
            **final_config,
            "use_instruction": self.use_instruction,
            "aspect_similarity_threshold": self.aspect_similarity_threshold,
            "two_stage_clustering": True
        }
        clustering_config_id = json.dumps(final_config_with_meta, sort_keys=True)
        
        # 插入issue_clusters
        for idx, (sentence_id, aspect_norm, issue_norm, sentiment, label) in enumerate(
            zip(all_sentence_ids, all_aspect_norms, all_issue_norms, all_sentiments, all_labels)
        ):
            is_noise = bool(label == -1)
            cluster_id = str(label) if not is_noise else "noise"
            cluster_key_text = self._build_cluster_text(aspect_norm, issue_norm)
            
            insert_query = f"""
                INSERT INTO {table_manager.ISSUE_CLUSTERS}
                (run_id, pipeline_version, data_slice_id, created_at,
                 embedding_model, clustering_config_id,
                 aspect_norm, cluster_id, sentence_id,
                 cluster_key_text, issue_norm, sentiment, is_noise)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.db.execute_write(insert_query, {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id,
                "created_at": self.version_fields["created_at"],
                "embedding_model": self.embedding_model,
                "clustering_config_id": clustering_config_id,
                "aspect_norm": aspect_norm,
                "cluster_id": cluster_id,
                "sentence_id": sentence_id,
                "cluster_key_text": cluster_key_text,
                "issue_norm": issue_norm,
                "sentiment": sentiment,
                "is_noise": is_noise
            })
        
        logger.info(f"已插入 {len(all_sentence_ids)} 条聚类归属记录")
        
        # 插入cluster_stats
        for cluster_id, stats in cluster_stats.items():
            insert_query = f"""
                INSERT INTO {table_manager.CLUSTER_STATS}
                (run_id, pipeline_version, data_slice_id, created_at,
                 aspect_norm, cluster_id, cluster_size, neg_ratio,
                 intra_cluster_distance, inter_cluster_distance, separation_ratio,
                 cohesion, cluster_confidence, sentiment_consistency,
                 representative_sentence_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            main_aspect_norm = stats.get("aspect_norm", "")
            medoid_sentence_id = stats.get("medoid_sentence_id", "")
            representative_ids = stats.get("representative_sentence_ids", [])
            
            # 将medoid添加到representative_ids的开头
            if medoid_sentence_id and medoid_sentence_id not in representative_ids:
                representative_ids = [medoid_sentence_id] + representative_ids[:19]  # 最多20个
            
            self.db.execute_write(insert_query, {
                "run_id": self.run_id,
                "pipeline_version": self.pipeline_version,
                "data_slice_id": self.data_slice_id,
                "created_at": self.version_fields["created_at"],
                "aspect_norm": main_aspect_norm,
                "cluster_id": str(cluster_id),
                "cluster_size": stats["cluster_size"],
                "neg_ratio": stats["neg_ratio"],
                "intra_cluster_distance": stats.get("intra_cluster_distance"),
                "inter_cluster_distance": stats.get("inter_cluster_distance"),
                "separation_ratio": stats.get("separation_ratio"),
                "cohesion": stats.get("cohesion"),
                "cluster_confidence": stats.get("cluster_confidence"),
                "sentiment_consistency": stats.get("sentiment_consistency"),
                "representative_sentence_ids": json.dumps(representative_ids)
            })
        
        logger.info(f"已插入 {len(cluster_stats)} 条簇统计记录")
        logger.info(
            f"IssueClusterAgent完成: 簇数={len(cluster_stats)}, "
            f"总记录数={len(all_sentence_ids)}"
        )
        
        return {
            "status": "success",
            "cluster_count": len(cluster_stats),
            "table": table_manager.ISSUE_CLUSTERS
        }
