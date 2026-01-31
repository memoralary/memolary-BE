"""
Knowledge Node Clustering Service

sentence-transformers를 사용하여 노드의 description을 벡터화하고,
코사인 유사도 기반 클러스터링을 수행합니다.

사용법:
    from services.knowledge.clustering import ClusteringService
    
    service = ClusteringService()
    
    # 단일 노드 클러스터 할당
    cluster_id, embedding = service.assign_cluster(
        description="딥러닝은 인공 신경망을 사용한 학습 방법이다."
    )
    
    # 배치 클러스터링
    results = service.assign_clusters_batch(nodes)
"""

import logging
import pickle
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import uuid

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

Vector = NDArray[np.float32]


# =============================================================================
# Clustering Service
# =============================================================================

@dataclass
class ClusterInfo:
    """클러스터 정보"""
    cluster_id: str
    centroid: Vector
    member_count: int = 0
    
    def update_centroid(self, new_vector: Vector) -> None:
        """새 벡터를 추가하여 중심점 업데이트 (온라인 평균)"""
        self.member_count += 1
        # 온라인 평균 업데이트: new_mean = old_mean + (new_value - old_mean) / n
        self.centroid = self.centroid + (new_vector - self.centroid) / self.member_count


@dataclass 
class ClusterAssignment:
    """클러스터 할당 결과"""
    cluster_id: str
    embedding: Vector
    similarity: float
    is_new_cluster: bool
    
    def get_embedding_bytes(self) -> bytes:
        """임베딩을 바이트로 직렬화 (DB 저장용)"""
        return pickle.dumps(self.embedding)


class ClusteringService:
    """
    노드 임베딩 및 클러스터링 서비스
    
    sentence-transformers를 사용하여 텍스트를 벡터화하고,
    코사인 유사도 기반으로 클러스터를 할당합니다.
    
    Args:
        model_name: sentence-transformers 모델명
        similarity_threshold: 기존 클러스터에 할당할 최소 유사도 (기본: 0.7)
        
    Example:
        service = ClusteringService()
        
        # 클러스터 할당
        result = service.assign_cluster("딥러닝은 신경망 기반 학습 방법이다.")
        print(f"클러스터: {result.cluster_id}, 유사도: {result.similarity}")
    """
    
    # 추천 모델들 (성능/속도 트레이드오프)
    RECOMMENDED_MODELS = {
        "fast": "paraphrase-MiniLM-L3-v2",           # 빠름, 저품질
        "balanced": "paraphrase-MiniLM-L6-v2",       # 균형
        "quality": "all-MiniLM-L6-v2",               # 품질 좋음
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 다국어
    }
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.7,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # Lazy loading
        self._model = None
        self._embedding_dim: Optional[int] = None
        
        # 클러스터 저장소 (메모리 기반, 실제 서비스에서는 DB 사용)
        self._clusters: Dict[str, ClusterInfo] = {}
    
    @property
    def model(self):
        """Lazy loading of sentence-transformer model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"모델 로딩 중: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"모델 로딩 완료: dim={self._embedding_dim}")
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers 패키지가 필요합니다. "
                    "'pip install sentence-transformers'를 실행하세요."
                )
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """임베딩 차원"""
        if self._embedding_dim is None:
            _ = self.model  # 모델 로딩 트리거
        return self._embedding_dim
    
    # =========================================================================
    # Embedding Methods
    # =========================================================================
    
    def encode(self, text: str) -> Vector:
        """
        텍스트를 벡터로 인코딩합니다.
        
        Args:
            text: 인코딩할 텍스트
            
        Returns:
            정규화된 임베딩 벡터
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 정규화 (코사인 유사도 계산 최적화)
        )
        return embedding.astype(np.float32)
    
    def encode_batch(self, texts: List[str]) -> Vector:
        """
        여러 텍스트를 배치로 인코딩합니다.
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            (N, embedding_dim) 형태의 임베딩 행렬
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        # 빈 텍스트 처리
        processed_texts = [t if t and t.strip() else " " for t in texts]
        
        embeddings = self.model.encode(
            processed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10
        )
        return embeddings.astype(np.float32)
    
    # =========================================================================
    # Similarity Methods
    # =========================================================================
    
    @staticmethod
    def cosine_similarity(vec1: Vector, vec2: Vector) -> float:
        """
        두 벡터 간 코사인 유사도 계산
        
        Note: 벡터가 이미 정규화되어 있다고 가정 (dot product = cosine similarity)
        """
        return float(np.dot(vec1, vec2))
    
    @staticmethod
    def cosine_similarity_matrix(
        query_vectors: Vector,
        corpus_vectors: Vector
    ) -> Vector:
        """
        쿼리 벡터들과 코퍼스 벡터들 간의 유사도 행렬 계산
        
        Args:
            query_vectors: (M, D) 쿼리 벡터 행렬
            corpus_vectors: (N, D) 코퍼스 벡터 행렬
            
        Returns:
            (M, N) 유사도 행렬
        """
        # 정규화된 벡터이므로 dot product = cosine similarity
        return np.dot(query_vectors, corpus_vectors.T)
    
    # =========================================================================
    # Cluster Management
    # =========================================================================
    
    def load_clusters_from_db(self, nodes_with_clusters: List[Dict]) -> None:
        """
        DB에서 기존 클러스터 정보를 로드하여 중심점 계산
        
        Args:
            nodes_with_clusters: [{"cluster_id": "...", "embedding": bytes}, ...]
        """
        cluster_embeddings: Dict[str, List[Vector]] = {}
        
        for node in nodes_with_clusters:
            cluster_id = node.get("cluster_id")
            embedding_bytes = node.get("embedding")
            
            if not cluster_id or not embedding_bytes:
                continue
            
            try:
                embedding = pickle.loads(embedding_bytes)
                if cluster_id not in cluster_embeddings:
                    cluster_embeddings[cluster_id] = []
                cluster_embeddings[cluster_id].append(embedding)
            except Exception as e:
                logger.warning(f"임베딩 로드 실패: {e}")
        
        # 중심점 계산
        for cluster_id, embeddings in cluster_embeddings.items():
            centroid = np.mean(embeddings, axis=0).astype(np.float32)
            self._clusters[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                centroid=centroid,
                member_count=len(embeddings)
            )
        
        logger.info(f"클러스터 {len(self._clusters)}개 로드됨")
    
    def add_cluster(self, cluster_id: str, centroid: Vector) -> None:
        """새 클러스터 추가"""
        self._clusters[cluster_id] = ClusterInfo(
            cluster_id=cluster_id,
            centroid=centroid,
            member_count=1
        )
    
    def get_cluster_centroids(self) -> Tuple[List[str], Vector]:
        """
        모든 클러스터의 중심점 반환
        
        Returns:
            (클러스터 ID 리스트, 중심점 행렬)
        """
        if not self._clusters:
            return [], np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        cluster_ids = list(self._clusters.keys())
        centroids = np.array([
            self._clusters[cid].centroid for cid in cluster_ids
        ], dtype=np.float32)
        
        return cluster_ids, centroids
    
    def _generate_cluster_id(self) -> str:
        """새 클러스터 ID 생성"""
        return f"cluster_{uuid.uuid4().hex[:8]}"
    
    # =========================================================================
    # Main Assignment Methods
    # =========================================================================
    
    def assign_cluster(
        self,
        description: str,
        node_title: Optional[str] = None,
    ) -> ClusterAssignment:
        """
        단일 노드에 클러스터를 할당합니다.
        
        1. description을 벡터화
        2. 기존 클러스터 중심점들과 코사인 유사도 계산
        3. 최대 유사도가 임계값 이상이면 해당 클러스터에 할당
        4. 미달이면 새 클러스터 생성
        
        Args:
            description: 노드 설명
            node_title: 노드 제목 (로깅용)
            
        Returns:
            ClusterAssignment: 할당된 클러스터 정보
        """
        # 텍스트 결합 (제목 + 설명)
        text = f"{node_title}: {description}" if node_title else description
        
        # 임베딩 생성
        embedding = self.encode(text)
        
        # 기존 클러스터가 없으면 새 클러스터 생성
        if not self._clusters:
            cluster_id = self._generate_cluster_id()
            self.add_cluster(cluster_id, embedding)
            
            logger.info(f"첫 번째 클러스터 생성: {cluster_id}")
            return ClusterAssignment(
                cluster_id=cluster_id,
                embedding=embedding,
                similarity=1.0,
                is_new_cluster=True
            )
        
        # 기존 클러스터 중심점들과 유사도 계산
        cluster_ids, centroids = self.get_cluster_centroids()
        similarities = self.cosine_similarity_matrix(
            embedding.reshape(1, -1),
            centroids
        )[0]
        
        # 최대 유사도 찾기
        max_idx = np.argmax(similarities)
        max_similarity = float(similarities[max_idx])
        best_cluster_id = cluster_ids[max_idx]
        
        # 임계값 비교
        if max_similarity >= self.similarity_threshold:
            # 기존 클러스터에 할당
            self._clusters[best_cluster_id].update_centroid(embedding)
            
            logger.debug(
                f"기존 클러스터 할당: {best_cluster_id} "
                f"(유사도: {max_similarity:.3f})"
            )
            return ClusterAssignment(
                cluster_id=best_cluster_id,
                embedding=embedding,
                similarity=max_similarity,
                is_new_cluster=False
            )
        else:
            # 새 클러스터 생성
            new_cluster_id = self._generate_cluster_id()
            self.add_cluster(new_cluster_id, embedding)
            
            logger.info(
                f"새 클러스터 생성: {new_cluster_id} "
                f"(최대 유사도: {max_similarity:.3f} < {self.similarity_threshold})"
            )
            return ClusterAssignment(
                cluster_id=new_cluster_id,
                embedding=embedding,
                similarity=max_similarity,
                is_new_cluster=True
            )
    
    def assign_clusters_batch(
        self,
        nodes: List[Dict[str, str]],
    ) -> List[ClusterAssignment]:
        """
        여러 노드에 배치로 클러스터를 할당합니다.
        
        Args:
            nodes: [{"title": "...", "description": "..."}, ...]
            
        Returns:
            ClusterAssignment 리스트
        """
        if not nodes:
            return []
        
        results = []
        
        # 텍스트 준비
        texts = [
            f"{n.get('title', '')}: {n.get('description', '')}"
            for n in nodes
        ]
        
        # 배치 임베딩
        logger.info(f"{len(nodes)}개 노드 임베딩 생성 중...")
        embeddings = self.encode_batch(texts)
        
        # 순차적으로 클러스터 할당 (온라인 클러스터링)
        for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
            if not self._clusters:
                # 첫 번째 노드는 무조건 새 클러스터
                cluster_id = self._generate_cluster_id()
                self.add_cluster(cluster_id, embedding)
                
                results.append(ClusterAssignment(
                    cluster_id=cluster_id,
                    embedding=embedding,
                    similarity=1.0,
                    is_new_cluster=True
                ))
            else:
                # 기존 클러스터들과 비교
                cluster_ids, centroids = self.get_cluster_centroids()
                similarities = np.dot(embedding, centroids.T)
                
                max_idx = np.argmax(similarities)
                max_similarity = float(similarities[max_idx])
                
                if max_similarity >= self.similarity_threshold:
                    best_cluster_id = cluster_ids[max_idx]
                    self._clusters[best_cluster_id].update_centroid(embedding)
                    
                    results.append(ClusterAssignment(
                        cluster_id=best_cluster_id,
                        embedding=embedding,
                        similarity=max_similarity,
                        is_new_cluster=False
                    ))
                else:
                    new_cluster_id = self._generate_cluster_id()
                    self.add_cluster(new_cluster_id, embedding)
                    
                    results.append(ClusterAssignment(
                        cluster_id=new_cluster_id,
                        embedding=embedding,
                        similarity=max_similarity,
                        is_new_cluster=True
                    ))
        
        # 결과 요약 로깅
        new_clusters = sum(1 for r in results if r.is_new_cluster)
        logger.info(
            f"배치 클러스터링 완료: {len(results)}개 노드, "
            f"{new_clusters}개 새 클러스터, "
            f"총 {len(self._clusters)}개 클러스터"
        )
        
        return results
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def find_similar_nodes(
        self,
        query_embedding: Vector,
        node_embeddings: List[Tuple[str, Vector]],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        쿼리 임베딩과 가장 유사한 노드들을 찾습니다.
        
        Args:
            query_embedding: 쿼리 벡터
            node_embeddings: [(node_id, embedding), ...]
            top_k: 반환할 상위 K개
            
        Returns:
            [(node_id, similarity), ...] 유사도 내림차순
        """
        if not node_embeddings:
            return []
        
        node_ids = [nid for nid, _ in node_embeddings]
        embeddings = np.array([emb for _, emb in node_embeddings])
        
        similarities = np.dot(query_embedding, embeddings.T)
        
        # Top-K 인덱스
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            (node_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """클러스터 통계 반환"""
        if not self._clusters:
            return {"num_clusters": 0}
        
        member_counts = [c.member_count for c in self._clusters.values()]
        
        return {
            "num_clusters": len(self._clusters),
            "total_members": sum(member_counts),
            "avg_members_per_cluster": np.mean(member_counts),
            "max_cluster_size": max(member_counts),
            "min_cluster_size": min(member_counts),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_service: Optional[ClusteringService] = None


def get_clustering_service(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold: float = 0.7,
) -> ClusteringService:
    """
    싱글톤 클러스터링 서비스 반환
    
    Args:
        model_name: sentence-transformers 모델명
        similarity_threshold: 유사도 임계값
        
    Returns:
        ClusteringService 인스턴스
    """
    global _default_service
    
    if _default_service is None:
        _default_service = ClusteringService(
            model_name=model_name,
            similarity_threshold=similarity_threshold
        )
    
    return _default_service


def embed_and_cluster(
    description: str,
    title: Optional[str] = None,
    threshold: float = 0.7,
) -> ClusterAssignment:
    """
    간편한 임베딩 및 클러스터 할당 함수
    
    Args:
        description: 노드 설명
        title: 노드 제목
        threshold: 유사도 임계값
        
    Returns:
        ClusterAssignment
    """
    service = get_clustering_service(similarity_threshold=threshold)
    return service.assign_cluster(description, title)
