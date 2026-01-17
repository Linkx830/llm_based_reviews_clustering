# 新增或修改我时需要修改这个文件夹中的README.md文件
"""聚类工具"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from ..utils.logger import get_logger

try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None

logger = get_logger(__name__)


class ClusteringTool:
    """
    聚类工具
    
    职责：
    - 执行聚类算法（HDBSCAN/AgglomerativeClustering）
    - 计算簇统计指标
    - 计算轮廓系数
    """
    
    def __init__(self, method: str = "hdbscan", metric: str = "cosine", **kwargs):
        """
        Args:
            method: 聚类方法 ('hdbscan', 'agglomerative', 'dbscan')
            metric: 距离度量（默认'cosine'，适用于已归一化的向量）
                - 'cosine': cosine距离（= 1 - cosine相似度）
                - 'euclidean': 欧氏距离
                - 'precomputed': 预计算距离矩阵
            **kwargs: 聚类参数
                - hdbscan: min_cluster_size, min_samples, metric
                - agglomerative: n_clusters (可选，如果不指定则自动确定), linkage, distance_threshold, metric
                - dbscan: eps, min_samples, metric
        """
        self.method = method
        self.metric = metric
        self.kwargs = kwargs
        self.model = None
        self.labels_ = None
    
    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """
        执行聚类
        
        Args:
            vectors: 特征向量 (n_samples, n_features)
        
        Returns:
            簇标签数组
        """
        if self.method == "hdbscan":
            if HDBSCAN is None:
                raise ImportError("hdbscan package is required for HDBSCAN clustering")
            min_cluster_size = self.kwargs.get("min_cluster_size", 5)
            min_samples = self.kwargs.get("min_samples", 3)
            metric = self.kwargs.get("metric", self.metric)
            
            # HDBSCAN的BallTree不支持'cosine'字符串，需要转换为可调用函数或使用'euclidean'
            # 由于向量已归一化，使用'euclidean'等价于cosine距离（在归一化向量上）
            if metric == "cosine":
                # 对于归一化向量，euclidean距离与cosine距离等价（仅比例不同）
                # euclidean^2 = 2 * (1 - cosine_similarity) = 2 * cosine_distance
                # 这不会影响聚类结果，因为距离的相对顺序保持不变
                logger.debug("HDBSCAN: 将'cosine'指标转换为'euclidean'（向量已归一化，结果等价）")
                metric = "euclidean"
            
            self.model = HDBSCAN(
                min_cluster_size=min_cluster_size, 
                min_samples=min_samples,
                metric=metric
            )
        elif self.method == "agglomerative":
            # AgglomerativeClustering 支持两种模式：
            # 1. 指定 n_clusters：固定簇数
            # 2. 指定 distance_threshold：基于距离阈值自动确定簇数
            # 如果两者都未指定，使用 distance_threshold 模式
            n_clusters = self.kwargs.get("n_clusters", None)
            distance_threshold = self.kwargs.get("distance_threshold", None)
            linkage = self.kwargs.get("linkage", "average")  # 使用average而不是ward（ward只支持euclidean）
            # 注意：新版本scikit-learn使用metric而不是affinity
            metric = self.kwargs.get("metric", self.metric)
            
            # 如果使用ward linkage，必须使用euclidean距离
            if linkage == "ward" and metric != "euclidean":
                logger.warning("ward linkage requires euclidean distance, switching to euclidean")
                metric = "euclidean"
            
            if n_clusters is not None:
                self.model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    metric=metric
                )
            elif distance_threshold is not None:
                self.model = AgglomerativeClustering(
                    distance_threshold=distance_threshold,
                    linkage=linkage,
                    metric=metric,
                    n_clusters=None
                )
            else:
                # 默认使用 distance_threshold 模式，自动确定簇数
                # 使用默认阈值（基于数据规模自适应）
                default_threshold = 0.5  # 可以根据实际情况调整
                self.model = AgglomerativeClustering(
                    distance_threshold=default_threshold,
                    linkage=linkage,
                    metric=metric,
                    n_clusters=None
                )
        elif self.method == "dbscan":
            eps = self.kwargs.get("eps", 0.5)
            min_samples = self.kwargs.get("min_samples", 5)
            metric = self.kwargs.get("metric", self.metric)
            self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        self.labels_ = self.model.fit_predict(vectors)
        return self.labels_
    
    def compute_silhouette(self, vectors: np.ndarray, labels: np.ndarray) -> float:
        """
        计算轮廓系数
        
        Args:
            vectors: 特征向量
            labels: 簇标签
        
        Returns:
            轮廓系数
        """
        # 过滤噪声点（标签为-1）
        mask = labels >= 0
        if mask.sum() < 2:
            return -1.0
        
        valid_vectors = vectors[mask]
        valid_labels = labels[mask]
        
        if len(np.unique(valid_labels)) < 2:
            return -1.0
        
        try:
            return silhouette_score(valid_vectors, valid_labels)
        except:
            return -1.0
    
    def compute_cluster_stats(
        self, 
        labels: np.ndarray, 
        vectors: np.ndarray = None,
        sentiments: List[str] = None,
        sample_ids: List[str] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        计算簇统计信息
        
        Args:
            labels: 簇标签
            vectors: 特征向量（可选，用于计算距离指标）
            sentiments: 情感标签列表（可选）
            sample_ids: 样本ID列表（可选）
        
        Returns:
            簇统计字典 {cluster_id: stats}
        """
        stats = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # 噪声点
                continue
            
            mask = labels == cluster_id
            cluster_size = mask.sum()
            
            stat = {
                "cluster_size": int(cluster_size),
                "neg_ratio": 0.0,
            }
            
            # 计算簇内距离和簇间距离（如果提供了向量）
            if vectors is not None:
                if cluster_size > 1:
                    cluster_vectors = vectors[mask]
                    # 计算簇中心
                    cluster_center = np.mean(cluster_vectors, axis=0)
                    # 计算簇内平均距离（到中心的平均距离）
                    intra_distances = np.linalg.norm(cluster_vectors - cluster_center, axis=1)
                    stat["intra_cluster_distance"] = float(np.mean(intra_distances))
                    stat["intra_cluster_std"] = float(np.std(intra_distances))
                    
                    # 计算簇间最小距离（到最近簇的距离）
                    inter_distances = []
                    for other_cluster_id in unique_labels:
                        if other_cluster_id == -1 or other_cluster_id == cluster_id:
                            continue
                        other_mask = labels == other_cluster_id
                        if other_mask.sum() > 0:
                            if other_mask.sum() > 1:
                                other_center = np.mean(vectors[other_mask], axis=0)
                            else:
                                # 单样本簇，直接使用该样本作为中心
                                other_center = vectors[other_mask][0]
                            distance = np.linalg.norm(cluster_center - other_center)
                            inter_distances.append(distance)
                    
                    if inter_distances:
                        stat["inter_cluster_distance"] = float(np.min(inter_distances))
                        # 处理簇内距离为0或非常接近0的情况（完全相同的样本）
                        EPSILON = 1e-6
                        MIN_INTRA_DISTANCE = 0.001  # 最小簇内距离阈值（用于计算分离度比率）
                        
                        if stat["intra_cluster_distance"] < MIN_INTRA_DISTANCE:
                            # 簇内距离过小，说明样本几乎完全相同
                            # 使用簇间距离的某个比例作为参考，而不是除以一个很小的值
                            # 分离度比率 = inter / (inter * 0.1) = 10，表示"簇间距离远大于簇内距离"
                            # 或者使用一个固定的合理值
                            if stat["inter_cluster_distance"] > 0:
                                # 使用簇间距离的10%作为参考簇内距离
                                reference_intra = stat["inter_cluster_distance"] * 0.1
                                stat["separation_ratio"] = float(stat["inter_cluster_distance"] / reference_intra)
                                logger.debug(
                                    f"簇 {cluster_id} 的簇内距离 {stat['intra_cluster_distance']:.6f} 过小，"
                                    f"使用参考值 {reference_intra:.6f} 计算分离度比率: {stat['separation_ratio']:.2f}"
                                )
                            else:
                                stat["separation_ratio"] = 10.0  # 默认值
                        else:
                            # 正常情况
                            stat["separation_ratio"] = float(stat["inter_cluster_distance"] / stat["intra_cluster_distance"])
                        
                        # 对异常大的值进行截断（通常分离度比率应该 < 1000）
                        MAX_SEPARATION_RATIO = 1000.0
                        if stat["separation_ratio"] > MAX_SEPARATION_RATIO:
                            logger.debug(
                                f"簇 {cluster_id} 的分离度比率 {stat['separation_ratio']:.2f} "
                                f"超过阈值 {MAX_SEPARATION_RATIO}，已截断"
                            )
                            stat["separation_ratio"] = MAX_SEPARATION_RATIO
                    
                    # 计算簇的紧密度（cohesion）：簇内距离的倒数
                    # 处理簇内距离为0或非常接近0的情况
                    MIN_INTRA_DISTANCE = 0.001  # 最小簇内距离阈值
                    
                    if stat["intra_cluster_distance"] < MIN_INTRA_DISTANCE:
                        # 簇内距离过小，说明样本几乎完全相同
                        # 使用一个合理的固定值，表示"非常紧密"
                        stat["cohesion"] = 1000.0  # 表示非常紧密，但不是无穷大
                        logger.debug(
                            f"簇 {cluster_id} 的簇内距离 {stat['intra_cluster_distance']:.6f} 过小，"
                            f"紧密度设为固定值 {stat['cohesion']:.2f}"
                        )
                    else:
                        # 正常情况：紧密度 = 1 / 簇内距离
                        stat["cohesion"] = 1.0 / stat["intra_cluster_distance"]
                    
                    # 对异常大的值进行截断（通常紧密度应该 < 10000）
                    MAX_COHESION = 10000.0
                    if stat["cohesion"] > MAX_COHESION:
                        logger.debug(
                            f"簇 {cluster_id} 的紧密度 {stat['cohesion']:.2f} "
                            f"超过阈值 {MAX_COHESION}，已截断"
                        )
                        stat["cohesion"] = MAX_COHESION
                    
                    # 计算簇的置信度（基于簇内一致性和分离度）
                    # 置信度 = separation_ratio / (1 + separation_ratio) * (1 - normalized_intra_distance)
                    if "separation_ratio" in stat and stat["separation_ratio"] > 0:
                        # 归一化簇内距离（相对于所有簇的平均簇内距离）
                        all_intra_distances = []
                        for cid in unique_labels:
                            if cid == -1 or cid == cluster_id:
                                continue
                            c_mask = labels == cid
                            if c_mask.sum() > 1:
                                c_vectors = vectors[c_mask]
                                c_center = np.mean(c_vectors, axis=0)
                                c_distances = np.linalg.norm(c_vectors - c_center, axis=1)
                                all_intra_distances.append(np.mean(c_distances))
                        
                        if all_intra_distances:
                            avg_intra_distance = np.mean(all_intra_distances)
                            normalized_intra = stat["intra_cluster_distance"] / avg_intra_distance if avg_intra_distance > 0 else 1.0
                            # 置信度计算：分离度越高、簇内距离越小，置信度越高
                            stat["cluster_confidence"] = float(
                                (stat["separation_ratio"] / (1 + stat["separation_ratio"])) * 
                                (1.0 / (1.0 + normalized_intra))
                            )
                        else:
                            stat["cluster_confidence"] = float(stat["separation_ratio"] / (1 + stat["separation_ratio"]))
                    else:
                        stat["cluster_confidence"] = 0.5  # 默认置信度
                elif cluster_size == 1:
                    # 单样本簇：计算到最近簇的距离作为分离度指标
                    cluster_vector = vectors[mask][0]
                    inter_distances = []
                    for other_cluster_id in unique_labels:
                        if other_cluster_id == -1 or other_cluster_id == cluster_id:
                            continue
                        other_mask = labels == other_cluster_id
                        if other_mask.sum() > 0:
                            if other_mask.sum() > 1:
                                other_center = np.mean(vectors[other_mask], axis=0)
                            else:
                                other_center = vectors[other_mask][0]
                            distance = np.linalg.norm(cluster_vector - other_center)
                            inter_distances.append(distance)
                    
                    if inter_distances:
                        stat["inter_cluster_distance"] = float(np.min(inter_distances))
                        # 对于单样本簇，使用平均簇内距离作为参考（如果存在）
                        all_intra_distances = []
                        for cid in unique_labels:
                            if cid == -1 or cid == cluster_id:
                                continue
                            c_mask = labels == cid
                            if c_mask.sum() > 1:
                                c_vectors = vectors[c_mask]
                                c_center = np.mean(c_vectors, axis=0)
                                c_distances = np.linalg.norm(c_vectors - c_center, axis=1)
                                all_intra_distances.append(np.mean(c_distances))
                        
                        if all_intra_distances:
                            avg_intra_distance = np.mean(all_intra_distances)
                            # 使用平均簇内距离作为参考，计算分离度比率
                            # 添加最小阈值保护
                            MIN_INTRA_DISTANCE = 0.001
                            if avg_intra_distance < MIN_INTRA_DISTANCE:
                                # 如果平均簇内距离也很小，使用簇间距离的10%作为参考
                                if stat["inter_cluster_distance"] > 0:
                                    reference_intra = stat["inter_cluster_distance"] * 0.1
                                    stat["separation_ratio"] = float(stat["inter_cluster_distance"] / reference_intra)
                                else:
                                    stat["separation_ratio"] = 10.0
                            else:
                                stat["separation_ratio"] = float(stat["inter_cluster_distance"] / avg_intra_distance)
                            
                            # 对异常大的值进行截断（使用统一的常量）
                            MAX_SEPARATION_RATIO = 1000.0
                            if stat["separation_ratio"] > MAX_SEPARATION_RATIO:
                                logger.debug(
                                    f"单样本簇 {cluster_id} 的分离度比率 {stat['separation_ratio']:.2f} "
                                    f"超过阈值 {MAX_SEPARATION_RATIO}，已截断"
                                )
                                stat["separation_ratio"] = MAX_SEPARATION_RATIO
                            # 单样本簇的置信度：基于分离度比率
                            stat["cluster_confidence"] = float(stat["separation_ratio"] / (1 + stat["separation_ratio"])) if stat["separation_ratio"] > 0 else 0.3
                        else:
                            # 如果所有簇都是单样本，使用默认值
                            stat["cluster_confidence"] = 0.3
                            stat["separation_ratio"] = 0.0
                    else:
                        # 只有一个簇，无法计算分离度
                        stat["cluster_confidence"] = 0.3
                        stat["separation_ratio"] = 0.0
                    
                    # 单样本簇的簇内距离设为0（只有一个点）
                    stat["intra_cluster_distance"] = 0.0
                    # 单样本簇紧密度设为固定值（避免数值爆炸）
                    # 使用一个合理的固定值，而不是1.0（因为1.0/infinity会导致问题）
                    stat["cohesion"] = 100.0  # 单样本簇紧密度设为固定值（表示最紧密）
                    
                    # 确保单样本簇的separation_ratio也被截断（如果之前计算过）
                    MAX_SEPARATION_RATIO = 1000.0
                    if "separation_ratio" in stat and stat["separation_ratio"] > MAX_SEPARATION_RATIO:
                        logger.debug(
                            f"单样本簇 {cluster_id} 的分离度比率 {stat['separation_ratio']:.2f} "
                            f"超过阈值 {MAX_SEPARATION_RATIO}，已截断"
                        )
                        stat["separation_ratio"] = MAX_SEPARATION_RATIO
            
            if sentiments:
                cluster_sentiments = [sentiments[i] for i in range(len(sentiments)) if mask[i]]
                neg_count = sum(1 for s in cluster_sentiments if s in ["negative", "neg"])
                stat["neg_ratio"] = neg_count / cluster_size if cluster_size > 0 else 0.0
                # 计算情感一致性（情感分布的熵）
                from collections import Counter
                sentiment_counts = Counter(cluster_sentiments)
                total = len(cluster_sentiments)
                if total > 0:
                    entropy = -sum((count/total) * np.log2(count/total) for count in sentiment_counts.values())
                    max_entropy = np.log2(len(sentiment_counts)) if len(sentiment_counts) > 1 else 1.0
                    stat["sentiment_consistency"] = float(1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0)
                else:
                    stat["sentiment_consistency"] = 0.0
            
            if sample_ids:
                cluster_samples = [sample_ids[i] for i in range(len(sample_ids)) if mask[i]]
                stat["representative_sentence_ids"] = cluster_samples[:20]  # 最多20个
            
            stats[int(cluster_id)] = stat
        
        return stats

