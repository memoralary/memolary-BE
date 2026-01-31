#!/usr/bin/env python
"""
Knowledge Graph 전체 파이프라인 사용 가이드

이 파일은 Django shell에서 단계별로 실행할 수 있습니다.

실행 방법:
1. 터미널에서:
   cd /Users/myeongsung/ET/backend
   source ../venv/bin/activate
   python manage.py shell

2. 아래 코드를 블록별로 복사하여 실행
"""

# =============================================================================
# 0. 환경 설정
# =============================================================================

from dotenv import load_dotenv
load_dotenv()

print("✅ 환경변수 로드 완료")


# =============================================================================
# 1단계: 텍스트에서 노드 추출 (LLM)
# =============================================================================

print("\n" + "=" * 60)
print("🔤 1단계: 텍스트에서 노드 추출")
print("=" * 60)

from services.knowledge.extractor import extract_nodes

# 예시 텍스트
sample_text = """
머신러닝(Machine Learning)은 인공지능의 한 분야로, 
데이터로부터 패턴을 학습하여 예측하는 알고리즘이다.

지도 학습(Supervised Learning)은 정답이 있는 데이터로 학습하고,
비지도 학습(Unsupervised Learning)은 정답 없이 패턴을 발견한다.

딥러닝(Deep Learning)은 심층 신경망을 사용하여 
더 복잡한 패턴을 학습할 수 있다.
"""

# 기존 노드 제목 (중복 제거용)
existing_titles = []  # 처음이면 빈 리스트

# 노드 추출
result = extract_nodes(sample_text, existing_titles)

print(f"추출된 노드: {result.unique_count}개")
for node in result.nodes:
    print(f"  📌 {node.title}: {node.description[:40]}...")


# =============================================================================
# 2단계: 노드에 임베딩 & 클러스터 할당
# =============================================================================

print("\n" + "=" * 60)
print("🔢 2단계: 노드 임베딩 & 클러스터링")
print("=" * 60)

from services.knowledge.clustering import ClusteringService

# 클러스터링 서비스 생성 (처음 실행 시 모델 다운로드)
clustering_service = ClusteringService(
    model_name='paraphrase-MiniLM-L6-v2',
    similarity_threshold=0.7
)

# 노드 데이터 준비
nodes_for_clustering = [
    {"title": node.title, "description": node.description or ""}
    for node in result.nodes
]

# 배치 클러스터링
cluster_results = clustering_service.assign_clusters_batch(nodes_for_clustering)

print("클러스터링 결과:")
for node, cluster in zip(result.nodes, cluster_results):
    status = "🆕 새 클러스터" if cluster.is_new_cluster else "📎 기존 클러스터"
    print(f"  {node.title} → {cluster.cluster_id[:15]}... {status}")


# =============================================================================
# 3단계: Django DB에 저장
# =============================================================================

print("\n" + "=" * 60)
print("💾 3단계: Django DB에 저장")
print("=" * 60)

from knowledge.models import KnowledgeNode

saved_nodes = []
for node, cluster in zip(result.nodes, cluster_results):
    # 이미 존재하는지 확인
    existing = KnowledgeNode.objects.filter(title=node.title).first()
    
    if existing:
        print(f"  ⏭️  {node.title} (이미 존재)")
        saved_nodes.append(existing)
    else:
        # 새 노드 생성
        new_node = KnowledgeNode.objects.create(
            title=node.title,
            description=node.description or "",
            cluster_id=cluster.cluster_id,
            tags=node.tags,
        )
        # 임베딩 저장
        new_node.set_embedding(cluster.embedding)
        new_node.save()
        
        print(f"  ✅ {node.title} 저장 완료 (ID: {str(new_node.id)[:8]}...)")
        saved_nodes.append(new_node)

print(f"\n총 저장된 노드: {len(saved_nodes)}개")


# =============================================================================
# 4단계: 선행조건 관계 생성 (LLM)
# =============================================================================

print("\n" + "=" * 60)
print("🔗 4단계: 선행조건 관계 생성")
print("=" * 60)

from services.knowledge.curriculum import generate_prerequisites

# 저장된 노드로 관계 생성
nodes_for_curriculum = [
    {"title": n.title, "description": n.description}
    for n in saved_nodes
]

prereq_result = generate_prerequisites(nodes_for_curriculum)

print(f"생성된 선행조건: {prereq_result.edge_count}개")
for edge in prereq_result.edges:
    print(f"  {edge.source} → {edge.target} (신뢰도: {edge.confidence})")

print(f"\n📚 권장 학습 순서:")
for i, title in enumerate(prereq_result.learning_order, 1):
    print(f"  {i}. {title}")


# =============================================================================
# 5단계: 엣지 DB에 저장
# =============================================================================

print("\n" + "=" * 60)
print("💾 5단계: 엣지 DB에 저장")
print("=" * 60)

from knowledge.models import KnowledgeEdge

# 제목 -> 노드 매핑
title_to_node = {n.title: n for n in saved_nodes}

saved_edges = []
for edge in prereq_result.edges:
    source_node = title_to_node.get(edge.source)
    target_node = title_to_node.get(edge.target)
    
    if not source_node or not target_node:
        print(f"  ⚠️  노드 없음: {edge.source} → {edge.target}")
        continue
    
    # 이미 존재하는지 확인
    existing = KnowledgeEdge.objects.filter(
        source=source_node,
        target=target_node,
        relation_type="prerequisite"
    ).first()
    
    if existing:
        print(f"  ⏭️  {edge.source} → {edge.target} (이미 존재)")
        saved_edges.append(existing)
    else:
        try:
            new_edge = KnowledgeEdge.objects.create(
                source=source_node,
                target=target_node,
                relation_type="prerequisite",
                confidence=edge.confidence,
                is_prerequisite=True,
            )
            print(f"  ✅ {edge.source} → {edge.target} 저장 완료")
            saved_edges.append(new_edge)
        except Exception as e:
            print(f"  ❌ 저장 실패: {e}")

print(f"\n총 저장된 엣지: {len(saved_edges)}개")


# =============================================================================
# 6단계: Link Prediction 모델 학습
# =============================================================================

print("\n" + "=" * 60)
print("🧠 6단계: Link Prediction 모델 학습")
print("=" * 60)

from services.knowledge.link_predictor import LinkPredictor, GraphDataLoader

# DB에서 그래프 로드
loader = GraphDataLoader()
graph_data = loader.load_from_db()

print(f"로드된 그래프: 노드 {graph_data.num_nodes}개, 엣지 {graph_data.num_edges}개")

if graph_data.num_edges >= 2:
    # 모델 학습
    predictor = LinkPredictor(
        embedding_dim=graph_data.embedding_dim,
        hidden_channels=64,
        out_channels=32,
    )
    
    history = predictor.train(
        graph_data,
        epochs=50,
        verbose=False
    )
    
    print(f"학습 완료! 초기 Loss: {history['train_loss'][0]:.4f} → 최종 Loss: {history['train_loss'][-1]:.4f}")
else:
    print("⚠️  엣지가 부족하여 학습 생략")
    predictor = None


# =============================================================================
# 7단계: 다음 학습 노드 예측
# =============================================================================

print("\n" + "=" * 60)
print("🔮 7단계: 다음 학습 노드 예측")
print("=" * 60)

if predictor and saved_nodes:
    # 첫 번째 노드를 현재 학습한 노드로 가정
    current_node = saved_nodes[0]
    current_node_id = str(current_node.id)
    
    print(f"현재 학습한 노드: {current_node.title}")
    print(f"\n추천 다음 학습 노드:")
    
    predictions = predictor.predict_next_nodes(
        current_node_id=current_node_id,
        top_k=3,
    )
    
    for rank, (node_id, title, score) in enumerate(predictions, 1):
        print(f"  {rank}. {title} (확률: {score:.1%})")
else:
    print("⚠️  예측 불가 (모델 없음 또는 노드 없음)")


# =============================================================================
# 🎉 완료!
# =============================================================================

print("\n" + "=" * 60)
print("🎉 전체 파이프라인 완료!")
print("=" * 60)

# 최종 통계
total_nodes = KnowledgeNode.objects.count()
total_edges = KnowledgeEdge.objects.count()

print(f"""
📊 최종 통계:
   총 노드: {total_nodes}개
   총 엣지: {total_edges}개
   
💡 다음 단계:
   1. 더 많은 텍스트로 노드 추가
   2. 모델 저장: predictor.save('model.pt')
   3. API 엔드포인트 생성
""")
