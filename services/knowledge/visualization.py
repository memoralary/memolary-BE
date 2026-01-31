"""
Galaxy Visualization Service - 3D 좌표 생성

UMAP을 사용하여 고차원 임베딩을 3D 좌표로 축소하고,
같은 클러스터의 노드들을 비슷한 영역에 배치합니다.

사용법:
    from services.knowledge.visualization import GalaxyVisualizer
    
    visualizer = GalaxyVisualizer()
    result = visualizer.generate_coordinates()
    
    # DB에 저장
    visualizer.save_to_db()
"""

import logging
import pickle
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NodeCoordinate:
    """노드 3D 좌표"""
    node_id: str
    title: str
    cluster_id: Optional[str]
    x: float
    y: float
    z: float


@dataclass
class VisualizationResult:
    """시각화 결과"""
    coordinates: List[NodeCoordinate]
    cluster_centers: Dict[str, Tuple[float, float, float]]
    dimensions: Tuple[float, float, float]  # (x_range, y_range, z_range)
    
    @property
    def num_nodes(self) -> int:
        return len(self.coordinates)
    
    @property
    def num_clusters(self) -> int:
        return len(self.cluster_centers)
    
    def to_dict_list(self) -> List[Dict]:
        """JSON 직렬화용"""
        return [
            {
                "node_id": c.node_id,
                "title": c.title,
                "cluster_id": c.cluster_id,
                "x": c.x,
                "y": c.y,
                "z": c.z,
            }
            for c in self.coordinates
        ]


# =============================================================================
# Galaxy Visualizer
# =============================================================================

class GalaxyVisualizer:
    """
    UMAP 기반 3D 은하수 좌표 생성기
    
    고차원 임베딩을 3D 좌표로 축소하고, 같은 클러스터의 노드들을
    비슷한 구(Sphere) 영역에 배치합니다.
    
    Example:
        visualizer = GalaxyVisualizer()
        result = visualizer.generate_coordinates()
        
        for coord in result.coordinates:
            print(f"{coord.title}: ({coord.x:.2f}, {coord.y:.2f}, {coord.z:.2f})")
        
        visualizer.save_to_db()
    """
    
    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        scale: float = 100.0,
        cluster_separation: float = 2.0,
        random_state: int = 42
    ):
        """
        Args:
            n_neighbors: UMAP 이웃 수 (클러스터링 강도, 높을수록 전역 구조 보존)
            min_dist: UMAP 최소 거리 (낮을수록 클러스터 밀집)
            spread: UMAP 확산 (클러스터 간 분리)
            scale: 좌표 스케일 (출력 좌표 범위)
            cluster_separation: 클러스터 간 추가 분리 거리
            random_state: 랜덤 시드
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.scale = scale
        self.cluster_separation = cluster_separation
        self.random_state = random_state
        
        self._umap = None
        self._result: Optional[VisualizationResult] = None
        self._node_data: List[Dict] = []
    
    @property
    def umap(self):
        """Lazy import of UMAP"""
        if self._umap is None:
            try:
                import umap
                self._umap = umap.UMAP(
                    n_components=3,
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    spread=self.spread,
                    random_state=self.random_state,
                    metric='cosine'
                )
            except ImportError:
                raise ImportError(
                    "umap-learn 패키지가 필요합니다. "
                    "'pip install umap-learn'을 실행하세요."
                )
        return self._umap
    
    def load_nodes_from_db(self) -> List[Dict]:
        """
        Django DB에서 노드와 임베딩 로드 (Raw SQL로 안전하게)
        """
        from django.db import connection
        
        nodes = []
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, embedding, cluster_id 
                FROM knowledge_knowledgenode
                WHERE embedding IS NOT NULL
            """)
            
            for row in cursor.fetchall():
                node_id = str(row[0])
                title = row[1]
                embedding_bytes = row[2]
                cluster_id = row[3]
                
                try:
                    if embedding_bytes:
                        if isinstance(embedding_bytes, memoryview):
                            embedding = pickle.loads(bytes(embedding_bytes))
                        else:
                            embedding = pickle.loads(embedding_bytes)
                        
                        nodes.append({
                            'id': node_id,
                            'title': title,
                            'embedding': embedding,
                            'cluster_id': cluster_id
                        })
                except Exception as e:
                    logger.warning(f"임베딩 로드 실패 ({title}): {e}")
        
        logger.info(f"DB에서 {len(nodes)}개 노드 로드됨")
        return nodes
    
    def generate_coordinates(
        self,
        nodes: Optional[List[Dict]] = None
    ) -> VisualizationResult:
        """
        3D 좌표 생성
        
        Args:
            nodes: [{"id": "...", "title": "...", "embedding": np.array, "cluster_id": "..."}]
                   None이면 DB에서 로드
        
        Returns:
            VisualizationResult
        """
        if nodes is None:
            nodes = self.load_nodes_from_db()
        
        if len(nodes) < 2:
            logger.warning("노드가 2개 미만이어서 3D 좌표를 생성할 수 없습니다.")
            return VisualizationResult(
                coordinates=[],
                cluster_centers={},
                dimensions=(0, 0, 0)
            )
        
        self._node_data = nodes
        
        # 임베딩 매트릭스 생성
        embeddings = np.array([n['embedding'] for n in nodes])
        
        # 노드가 적으면 n_neighbors 조정
        actual_n_neighbors = min(self.n_neighbors, len(nodes) - 1)
        if actual_n_neighbors < self.n_neighbors:
            logger.info(f"노드 수가 적어 n_neighbors를 {actual_n_neighbors}로 조정")
            self._umap = None  # 재생성
            import umap
            self._umap = umap.UMAP(
                n_components=3,
                n_neighbors=actual_n_neighbors,
                min_dist=self.min_dist,
                spread=self.spread,
                random_state=self.random_state,
                metric='cosine'
            )
        
        # UMAP 차원 축소
        logger.info("UMAP 3D 변환 중...")
        coords_3d = self.umap.fit_transform(embeddings)
        
        # 스케일 조정 및 정규화
        coords_3d = self._normalize_and_scale(coords_3d)
        
        # 클러스터별 중심점 계산 및 분리 강화
        coords_3d = self._enhance_cluster_separation(coords_3d, nodes)
        
        # 결과 생성
        coordinates = []
        for i, node in enumerate(nodes):
            coordinates.append(NodeCoordinate(
                node_id=node['id'],
                title=node['title'],
                cluster_id=node.get('cluster_id'),
                x=float(coords_3d[i, 0]),
                y=float(coords_3d[i, 1]),
                z=float(coords_3d[i, 2])
            ))
        
        # 클러스터 중심점 계산
        cluster_centers = self._calculate_cluster_centers(coordinates)
        
        # 차원 범위
        dimensions = (
            float(coords_3d[:, 0].max() - coords_3d[:, 0].min()),
            float(coords_3d[:, 1].max() - coords_3d[:, 1].min()),
            float(coords_3d[:, 2].max() - coords_3d[:, 2].min())
        )
        
        self._result = VisualizationResult(
            coordinates=coordinates,
            cluster_centers=cluster_centers,
            dimensions=dimensions
        )
        
        logger.info(f"3D 좌표 생성 완료: {len(coordinates)}개 노드, {len(cluster_centers)}개 클러스터")
        
        return self._result
    
    def _normalize_and_scale(self, coords: NDArray) -> NDArray:
        """좌표 정규화 및 스케일 조정"""
        # 중심을 원점으로
        coords = coords - coords.mean(axis=0)
        
        # 스케일 조정
        max_range = np.abs(coords).max()
        if max_range > 0:
            coords = coords / max_range * self.scale
        
        return coords
    
    def _enhance_cluster_separation(
        self,
        coords: NDArray,
        nodes: List[Dict]
    ) -> NDArray:
        """클러스터 간 분리 강화"""
        # 클러스터별 그룹화
        cluster_indices: Dict[str, List[int]] = {}
        for i, node in enumerate(nodes):
            cluster_id = node.get('cluster_id') or 'default'
            if cluster_id not in cluster_indices:
                cluster_indices[cluster_id] = []
            cluster_indices[cluster_id].append(i)
        
        if len(cluster_indices) <= 1:
            return coords
        
        # 각 클러스터의 중심점 계산
        cluster_centers = {}
        for cluster_id, indices in cluster_indices.items():
            cluster_coords = coords[indices]
            cluster_centers[cluster_id] = cluster_coords.mean(axis=0)
        
        # 클러스터 중심점들 간의 분리 강화
        center_coords = np.array(list(cluster_centers.values()))
        global_center = center_coords.mean(axis=0)
        
        # 각 클러스터를 중심에서 더 멀리 이동
        for cluster_id, indices in cluster_indices.items():
            center = cluster_centers[cluster_id]
            direction = center - global_center
            norm = np.linalg.norm(direction)
            
            if norm > 0:
                direction = direction / norm
                offset = direction * self.cluster_separation * self.scale / len(cluster_indices)
                coords[indices] += offset
        
        return coords
    
    def _calculate_cluster_centers(
        self,
        coordinates: List[NodeCoordinate]
    ) -> Dict[str, Tuple[float, float, float]]:
        """클러스터 중심점 계산"""
        cluster_coords: Dict[str, List[Tuple[float, float, float]]] = {}
        
        for coord in coordinates:
            cluster_id = coord.cluster_id or 'default'
            if cluster_id not in cluster_coords:
                cluster_coords[cluster_id] = []
            cluster_coords[cluster_id].append((coord.x, coord.y, coord.z))
        
        centers = {}
        for cluster_id, coords in cluster_coords.items():
            x = np.mean([c[0] for c in coords])
            y = np.mean([c[1] for c in coords])
            z = np.mean([c[2] for c in coords])
            centers[cluster_id] = (float(x), float(y), float(z))
        
        return centers
    
    def save_to_db(self) -> int:
        """
        생성된 3D 좌표를 Django DB에 저장
        
        Returns:
            업데이트된 노드 수
        """
        if self._result is None:
            raise ValueError("먼저 generate_coordinates()를 실행하세요.")
        
        from django.db import connection
        
        updated_count = 0
        
        with connection.cursor() as cursor:
            for coord in self._result.coordinates:
                try:
                    cursor.execute("""
                        UPDATE knowledge_knowledgenode
                        SET x = %s, y = %s, z = %s
                        WHERE id = %s
                    """, [coord.x, coord.y, coord.z, coord.node_id])
                    updated_count += cursor.rowcount
                except Exception as e:
                    logger.error(f"좌표 저장 실패 ({coord.title}): {e}")
        
        logger.info(f"DB에 {updated_count}개 노드 좌표 저장됨")
        return updated_count
    
    def get_json_export(self) -> Dict[str, Any]:
        """
        Three.js 등 프론트엔드용 JSON 내보내기
        """
        if self._result is None:
            raise ValueError("먼저 generate_coordinates()를 실행하세요.")
        
        return {
            "nodes": [
                {
                    "id": c.node_id,
                    "title": c.title,
                    "cluster": c.cluster_id,
                    "position": [c.x, c.y, c.z]
                }
                for c in self._result.coordinates
            ],
            "clusters": [
                {
                    "id": cluster_id,
                    "center": list(center)
                }
                for cluster_id, center in self._result.cluster_centers.items()
            ],
            "metadata": {
                "total_nodes": self._result.num_nodes,
                "total_clusters": self._result.num_clusters,
                "dimensions": {
                    "x_range": self._result.dimensions[0],
                    "y_range": self._result.dimensions[1],
                    "z_range": self._result.dimensions[2]
                }
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_galaxy_coordinates(
    scale: float = 100.0
) -> VisualizationResult:
    """
    간편한 은하수 좌표 생성 함수
    
    Returns:
        VisualizationResult
    """
    visualizer = GalaxyVisualizer(scale=scale)
    return visualizer.generate_coordinates()


def update_node_coordinates() -> int:
    """
    DB의 모든 노드에 3D 좌표 업데이트
    
    Returns:
        업데이트된 노드 수
    """
    visualizer = GalaxyVisualizer()
    visualizer.generate_coordinates()
    return visualizer.save_to_db()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌌 Galaxy Visualizer 테스트")
    print("=" * 60)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    
    test_nodes = [
        {"id": "1", "title": "머신러닝", "embedding": np.random.randn(64), "cluster_id": "ml"},
        {"id": "2", "title": "딥러닝", "embedding": np.random.randn(64), "cluster_id": "ml"},
        {"id": "3", "title": "신경망", "embedding": np.random.randn(64), "cluster_id": "ml"},
        {"id": "4", "title": "선형대수", "embedding": np.random.randn(64), "cluster_id": "math"},
        {"id": "5", "title": "미적분", "embedding": np.random.randn(64), "cluster_id": "math"},
        {"id": "6", "title": "통계학", "embedding": np.random.randn(64), "cluster_id": "math"},
    ]
    
    print(f"\n📊 테스트 노드: {len(test_nodes)}개")
    
    visualizer = GalaxyVisualizer(scale=50.0)
    result = visualizer.generate_coordinates(nodes=test_nodes)
    
    print(f"\n🌌 결과:")
    print(f"   노드 수: {result.num_nodes}")
    print(f"   클러스터 수: {result.num_clusters}")
    print(f"   차원 범위: x={result.dimensions[0]:.1f}, y={result.dimensions[1]:.1f}, z={result.dimensions[2]:.1f}")
    
    print(f"\n📍 좌표:")
    for coord in result.coordinates:
        print(f"   {coord.title:10} [{coord.cluster_id}]: ({coord.x:6.1f}, {coord.y:6.1f}, {coord.z:6.1f})")
    
    print(f"\n🎯 클러스터 중심:")
    for cluster_id, center in result.cluster_centers.items():
        print(f"   {cluster_id}: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    
    print("\n🎉 테스트 완료!")
