"""
KnowledgeNode와 KnowledgeEdge CRUD 테스트 스니펫
Django Shell에서 테스트: python manage.py shell

아래 코드를 한 블록씩 실행하면서 각 기능을 테스트하세요.
"""

# =============================================================================
# 0. 초기 설정
# =============================================================================
import numpy as np
from knowledge.models import KnowledgeNode, KnowledgeEdge
from django.core.exceptions import ValidationError


# =============================================================================
# 1. CREATE - 노드 생성
# =============================================================================

# 단일 노드 생성
node1 = KnowledgeNode.objects.create(
    title="Machine Learning",
    description="기계학습의 기본 개념과 알고리즘",
    cluster_id="ml_cluster_01",
    tags=["ml", "ai", "fundamentals"]
)
print(f"생성된 노드: {node1.title} (ID: {node1.id})")

# 임베딩 벡터와 함께 노드 생성
node2 = KnowledgeNode(
    title="Deep Learning",
    description="심층 신경망 기반 학습",
    cluster_id="ml_cluster_01",
    tags=["ml", "ai", "neural-network"]
)
# 임베딩 설정 (768차원 벡터 예시)
node2.set_embedding(np.random.randn(768).astype(np.float32))
node2.save()
print(f"임베딩과 함께 생성된 노드: {node2.title}")

# 추가 노드 생성 (엣지 테스트용)
node3 = KnowledgeNode.objects.create(
    title="Linear Algebra",
    description="선형대수학 기초",
    cluster_id="math_cluster_01",
    tags=["math", "fundamentals"]
)

node4 = KnowledgeNode.objects.create(
    title="Neural Networks",
    description="인공 신경망 아키텍처",
    cluster_id="ml_cluster_01",
    tags=["ml", "neural-network"]
)

print(f"총 노드 수: {KnowledgeNode.objects.count()}")


# =============================================================================
# 2. CREATE - 엣지 생성
# =============================================================================

# 기본 엣지 생성
edge1 = KnowledgeEdge.objects.create(
    source=node3,  # Linear Algebra
    target=node1,  # Machine Learning
    relation_type="prerequisite",
    confidence=0.95
)
print(f"생성된 엣지: {edge1}")
print(f"is_prerequisite 자동 설정: {edge1.is_prerequisite}")  # True

# 다른 관계 유형의 엣지
edge2 = KnowledgeEdge.objects.create(
    source=node1,  # Machine Learning
    target=node2,  # Deep Learning
    relation_type="related",
    confidence=0.85,
    is_prerequisite=False
)
print(f"생성된 엣지: {edge2}")

edge3 = KnowledgeEdge.objects.create(
    source=node2,  # Deep Learning
    target=node4,  # Neural Networks
    relation_type="includes",
    confidence=0.90
)
print(f"총 엣지 수: {KnowledgeEdge.objects.count()}")


# 자기 참조 엣지 생성 시도 (ValidationError 발생해야 함)
try:
    invalid_edge = KnowledgeEdge.objects.create(
        source=node1,
        target=node1,  # 동일한 노드!
        relation_type="related",
        confidence=1.0
    )
except ValidationError as e:
    print(f"예상된 ValidationError: {e}")


# 중복 엣지 생성 시도 (unique_together 위반)
try:
    duplicate_edge = KnowledgeEdge.objects.create(
        source=node3,
        target=node1,
        relation_type="prerequisite",  # 이미 존재하는 조합
        confidence=0.99
    )
except Exception as e:
    print(f"예상된 IntegrityError: {type(e).__name__}")


# =============================================================================
# 3. READ - 조회
# =============================================================================

# 전체 노드 조회
all_nodes = KnowledgeNode.objects.all()
print(f"전체 노드: {list(all_nodes.values_list('title', flat=True))}")

# UUID로 노드 조회
retrieved_node = KnowledgeNode.objects.get(id=node1.id)
print(f"ID로 조회: {retrieved_node.title}")

# title로 노드 조회 (unique 필드)
ml_node = KnowledgeNode.objects.get(title="Machine Learning")
print(f"Title로 조회: {ml_node.title}")

# 클러스터별 노드 조회
ml_cluster_nodes = KnowledgeNode.objects.filter(cluster_id="ml_cluster_01")
print(f"ML 클러스터 노드: {list(ml_cluster_nodes.values_list('title', flat=True))}")

# 태그로 노드 필터링 (JSONField)
ai_tagged_nodes = KnowledgeNode.objects.filter(tags__contains=["ai"])
print(f"AI 태그 노드: {list(ai_tagged_nodes.values_list('title', flat=True))}")

# 노드의 임베딩 읽기
embedding = node2.get_embedding()
if embedding is not None:
    print(f"임베딩 shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"임베딩 샘플 (처음 5개): {embedding[:5]}")

# 노드의 나가는 엣지 조회
out_edges = node1.out_edges.all()
print(f"{node1.title}의 나가는 엣지: {list(out_edges)}")

# 노드의 들어오는 엣지 조회
in_edges = node1.in_edges.all()
print(f"{node1.title}의 들어오는 엣지: {list(in_edges)}")

# 선행조건 엣지만 조회
prerequisite_edges = KnowledgeEdge.objects.filter(is_prerequisite=True)
print(f"선행조건 엣지: {list(prerequisite_edges)}")

# 특정 관계 유형의 엣지 조회
related_edges = KnowledgeEdge.objects.filter(relation_type="related")
print(f"Related 엣지: {list(related_edges)}")


# =============================================================================
# 4. UPDATE - 수정
# =============================================================================

# 노드 수정
node1.description = "Updated: 기계학습은 데이터로부터 학습하는 알고리즘 연구 분야"
node1.tags.append("updated")
node1.save()
print(f"수정된 노드: {node1.description}")
print(f"수정된 태그: {node1.tags}")

# 노드 임베딩 업데이트
new_embedding = np.random.randn(768).astype(np.float32)
node1.set_embedding(new_embedding)
node1.save()
print(f"임베딩 업데이트됨: {node1.get_embedding()[:5]}")

# 엣지 수정
edge1.confidence = 0.99
edge1.save()
print(f"수정된 엣지 confidence: {edge1.confidence}")

# bulk update
KnowledgeNode.objects.filter(cluster_id="ml_cluster_01").update(
    cluster_id="ml_cluster_v2"
)
print("클러스터 ID 일괄 업데이트 완료")


# =============================================================================
# 5. DELETE - 삭제
# =============================================================================

# 엣지 삭제
edge_to_delete = KnowledgeEdge.objects.get(id=edge3.id)
edge_to_delete.delete()
print(f"엣지 삭제됨. 남은 엣지 수: {KnowledgeEdge.objects.count()}")

# 노드 삭제 (CASCADE로 연결된 엣지도 함께 삭제됨)
node_to_delete = KnowledgeNode.objects.get(title="Neural Networks")
node_to_delete.delete()
print(f"노드 삭제됨. 남은 노드 수: {KnowledgeNode.objects.count()}")

# 조건부 삭제
KnowledgeNode.objects.filter(tags__contains=["test"]).delete()
print("테스트 태그 노드 삭제 완료")


# =============================================================================
# 6. 고급 쿼리 예시
# =============================================================================

# 특정 노드의 모든 선행조건 노드 가져오기
def get_prerequisites(node):
    """주어진 노드의 모든 선행조건 노드 반환"""
    prereq_edges = node.in_edges.filter(is_prerequisite=True)
    return [edge.source for edge in prereq_edges]

ml_prerequisites = get_prerequisites(node1)
print(f"{node1.title}의 선행조건: {[n.title for n in ml_prerequisites]}")


# 특정 노드에서 도달 가능한 모든 노드 (BFS)
def get_reachable_nodes(start_node, max_depth=3):
    """BFS로 도달 가능한 모든 노드 반환"""
    visited = set()
    queue = [(start_node, 0)]
    
    while queue:
        current, depth = queue.pop(0)
        if current.id in visited or depth > max_depth:
            continue
        visited.add(current.id)
        
        for edge in current.out_edges.all():
            queue.append((edge.target, depth + 1))
    
    return list(KnowledgeNode.objects.filter(id__in=visited))

reachable = get_reachable_nodes(node3)
print(f"{node3.title}에서 도달 가능한 노드: {[n.title for n in reachable]}")


# 클러스터별 노드 통계
from django.db.models import Count
cluster_stats = KnowledgeNode.objects.values('cluster_id').annotate(
    node_count=Count('id')
).order_by('-node_count')
print(f"클러스터별 통계: {list(cluster_stats)}")


# =============================================================================
# 7. GNN을 위한 데이터 변환
# =============================================================================

def get_graph_data_for_gnn():
    """GNN 학습을 위한 그래프 데이터 반환"""
    # 노드 목록 및 ID 매핑
    nodes = list(KnowledgeNode.objects.all())
    node_to_idx = {node.id: idx for idx, node in enumerate(nodes)}
    
    # 노드 특성 행렬 (임베딩 또는 원-핫)
    features = []
    for node in nodes:
        emb = node.get_embedding()
        if emb is not None:
            features.append(emb)
        else:
            features.append(np.zeros(768))
    features = np.stack(features)
    
    # 엣지 인덱스 (COO 형식: source_indices, target_indices)
    edges = KnowledgeEdge.objects.all()
    edge_index = np.array([
        [node_to_idx[e.source_id], node_to_idx[e.target_id]]
        for e in edges
    ]).T if edges.exists() else np.zeros((2, 0), dtype=np.int64)
    
    # 엣지 특성 (confidence, is_prerequisite, relation_type encoding)
    edge_attr = np.array([
        [e.confidence, float(e.is_prerequisite)]
        for e in edges
    ]) if edges.exists() else np.zeros((0, 2))
    
    return {
        'node_features': features,      # (num_nodes, 768)
        'edge_index': edge_index,        # (2, num_edges)
        'edge_attr': edge_attr,          # (num_edges, 2)
        'node_titles': [n.title for n in nodes],
    }

# 사용 예시
# graph_data = get_graph_data_for_gnn()
# print(f"노드 수: {graph_data['node_features'].shape[0]}")
# print(f"엣지 수: {graph_data['edge_index'].shape[1]}")


# =============================================================================
# 8. 정리 (선택적)
# =============================================================================

# 테스트 데이터 전체 삭제
# KnowledgeEdge.objects.all().delete()
# KnowledgeNode.objects.all().delete()
# print("모든 데이터 삭제 완료")
