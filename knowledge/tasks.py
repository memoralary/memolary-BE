"""
Celery 태스크 - Knowledge Graph Pipeline

백그라운드에서 실행되는 무거운 작업들
- 병렬 청크 처리로 PDF 분석 속도 개선
"""

import os
import logging
from typing import Dict, Any, Optional, List
from celery import shared_task, group, chord
from celery.exceptions import Ignore

logger = logging.getLogger(__name__)


# =============================================================================
# 청크별 노드 추출 태스크 (병렬 처리용)
# =============================================================================

@shared_task(bind=True, name='knowledge.tasks.extract_chunk_nodes')
def extract_chunk_nodes(
    self,
    chunk: str,
    chunk_index: int,
    existing_titles: List[str]
) -> Dict[str, Any]:
    """
    단일 청크에서 노드 추출 (병렬 실행용)
    
    Args:
        chunk: 텍스트 청크
        chunk_index: 청크 인덱스 (순서 보존용)
        existing_titles: 기존 노드 제목 목록 (중복 방지)
        
    Returns:
        추출된 노드 정보 딕셔너리
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        from services.knowledge.extractor import extract_nodes
        
        extraction_result = extract_nodes(chunk, existing_titles)
        
        # 직렬화 가능한 형태로 변환
        nodes_data = [
            {
                "title": node.title,
                "description": node.description or "",
                "tags": node.tags or []
            }
            for node in extraction_result.nodes
        ]
        
        logger.info(f"[청크 {chunk_index}] {len(nodes_data)}개 노드 추출 완료")
        
        return {
            "chunk_index": chunk_index,
            "nodes": nodes_data,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        logger.warning(f"[청크 {chunk_index}] 추출 실패: {e}")
        return {
            "chunk_index": chunk_index,
            "nodes": [],
            "success": False,
            "error": str(e)
        }


@shared_task(bind=True, name='knowledge.tasks.process_ingestion')
def process_ingestion(
    self,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    options: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    텍스트 또는 파일을 처리하여 Knowledge Graph 파이프라인 실행
    
    Args:
        text: 직접 입력된 텍스트
        file_path: 업로드된 파일 경로
        options: 추가 옵션
        
    Returns:
        처리 결과 (노드 수, 엣지 수 등)
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    options = options or {}
    result = {
        "status": "processing",
        "task_id": self.request.id,
        "steps": [],
        "nodes_created": 0,
        "edges_created": 0,
        "errors": []
    }
    
    try:
        # =================================================================
        # 1단계: 입력 처리 (Ingestion)
        # =================================================================
        self.update_state(state='PROGRESS', meta={'step': 1, 'message': 'Ingestion 처리 중...'})
        
        from services.knowledge.ingestion import IngestionService
        
        service = IngestionService(chunk_size=4000)
        
        if text:
            ingestion_result = service.process(text)
        elif file_path:
            ingestion_result = service.process(file_path)
        else:
            raise ValueError("텍스트 또는 파일이 필요합니다.")
        
        result["steps"].append({
            "step": 1,
            "name": "Ingestion",
            "success": True,
            "chunks": ingestion_result.chunk_count
        })
        
        # =================================================================
        # 2단계: 노드 추출 (병렬 처리)
        # =================================================================
        self.update_state(state='PROGRESS', meta={'step': 2, 'message': f'노드 추출 중... ({ingestion_result.chunk_count}개 청크 병렬 처리)'})
        
        from knowledge.models import KnowledgeNode
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        existing_titles = list(KnowledgeNode.objects.values_list('title', flat=True))
        
        # 병렬 처리 함수
        def process_chunk(args):
            chunk, chunk_idx = args
            from services.knowledge.extractor import extract_nodes
            try:
                extraction_result = extract_nodes(chunk, existing_titles)
                return {
                    "chunk_index": chunk_idx,
                    "nodes": [
                        {
                            "title": node.title,
                            "description": node.description or "",
                            "tags": node.tags or []
                        }
                        for node in extraction_result.nodes
                    ],
                    "success": True,
                    "error": None
                }
            except Exception as e:
                logger.warning(f"[청크 {chunk_idx}] 추출 실패: {e}")
                return {
                    "chunk_index": chunk_idx,
                    "nodes": [],
                    "success": False,
                    "error": str(e)
                }
        
        # ThreadPoolExecutor로 병렬 처리 (I/O 바운드 작업에 적합)
        max_workers = min(8, len(ingestion_result.chunks))  # 최대 8개 병렬
        chunk_args = [(chunk, idx) for idx, chunk in enumerate(ingestion_result.chunks)]
        
        all_nodes = []
        seen_titles = set(existing_titles)  # 중복 제거용
        
        logger.info(f"병렬 처리 시작: {len(chunk_args)}개 청크, {max_workers}개 워커")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, args): args[1] for args in chunk_args}
            
            for future in as_completed(futures):
                chunk_result = future.result()
                
                if chunk_result["success"]:
                    for node_data in chunk_result["nodes"]:
                        # 중복 제거
                        if node_data["title"] not in seen_titles:
                            seen_titles.add(node_data["title"])
                            all_nodes.append(node_data)
                else:
                    result["errors"].append(chunk_result["error"])
        
        logger.info(f"병렬 처리 완료: {len(all_nodes)}개 노드 추출됨")
        
        result["steps"].append({
            "step": 2,
            "name": "Node Extraction (Parallel)",
            "success": True,
            "nodes_extracted": len(all_nodes),
            "workers_used": max_workers
        })
        
        if not all_nodes:
            result["status"] = "completed"
            result["message"] = "추출된 노드가 없습니다."
            return result
        
        # =================================================================
        # 3단계: 클러스터링 & 임베딩
        # =================================================================
        self.update_state(state='PROGRESS', meta={'step': 3, 'message': '클러스터링 중...'})
        
        from services.knowledge.clustering import ClusteringService
        
        clustering_service = ClusteringService(
            model_name='paraphrase-MiniLM-L6-v2',
            similarity_threshold=0.7
        )
        
        nodes_for_clustering = [
            {"title": node["title"], "description": node["description"]}
            for node in all_nodes
        ]
        
        cluster_results = clustering_service.assign_clusters_batch(nodes_for_clustering)
        
        result["steps"].append({
            "step": 3,
            "name": "Clustering",
            "success": True,
            "clusters": len(set(c.cluster_id for c in cluster_results))
        })
        
        # =================================================================
        # 4단계: DB 저장
        # =================================================================
        self.update_state(state='PROGRESS', meta={'step': 4, 'message': 'DB 저장 중...'})
        
        from knowledge.models import TrackType
        
        # 사투리/방언 도메인 키워드 (TRACK_B)
        DIALECT_KEYWORDS = [
            '사투리', '방언', '경상도', '전라도', '억양', '민속',
            '무당', '택호', '호칭', '전남', '전북', '경북', '경남',
            '충청', '제주', '강원', '지역어', '토속어'
        ]
        
        def get_track_type(title: str, tags: list) -> str:
            """제목/태그 기반으로 track_type 결정"""
            text = (title + ' ' + ' '.join(tags or [])).lower()
            for keyword in DIALECT_KEYWORDS:
                if keyword in text:
                    return TrackType.TRACK_B
            return TrackType.TRACK_A
        
        saved_nodes = []
        for node, cluster in zip(all_nodes, cluster_results):
            existing = KnowledgeNode.objects.filter(title=node["title"]).first()
            
            if existing:
                if not existing.embedding:
                    existing.set_embedding(cluster.embedding)
                    existing.save(update_fields=['embedding'])
                saved_nodes.append(existing)
            else:
                # track_type 자동 분류
                from services.knowledge.track_classifier import classify_track_type
                track_type = classify_track_type(
                    title=node.title,
                    tags=node.tags,
                    description=node.description
                )
                
                new_node = KnowledgeNode.objects.create(
                    title=node["title"],
                    description=node["description"],
                    cluster_id=cluster.cluster_id,
                    tags=node.tags,
                    track_type=track_type,  # 자동 분류된 track_type 사용
                )
                new_node.set_embedding(cluster.embedding)
                new_node.save()
                saved_nodes.append(new_node)
                result["nodes_created"] += 1
                logger.info(f"노드 저장: {node['title']} (track={track_type})")
        
        result["steps"].append({
            "step": 4,
            "name": "DB Save",
            "success": True,
            "nodes_saved": len(saved_nodes)
        })
        
        # =================================================================
        # 5단계: 관계 생성
        # =================================================================
        self.update_state(state='PROGRESS', meta={'step': 5, 'message': '관계 추론 중...'})
        
        from services.knowledge.curriculum import generate_prerequisites
        from knowledge.models import KnowledgeEdge
        
        nodes_for_curriculum = [
            {"title": n.title, "description": n.description}
            for n in saved_nodes
        ]
        
        prereq_result = generate_prerequisites(nodes_for_curriculum)
        
        title_to_node = {n.title: n for n in saved_nodes}
        
        for edge in prereq_result.edges:
            source_node = title_to_node.get(edge.source)
            target_node = title_to_node.get(edge.target)
            
            if source_node and target_node:
                edge_obj, created = KnowledgeEdge.objects.get_or_create(
                    source=source_node,
                    target=target_node,
                    relation_type="prerequisite",
                    defaults={
                        "confidence": edge.confidence,
                        "is_prerequisite": True,
                    }
                )
                if created:
                    result["edges_created"] += 1
        
        result["steps"].append({
            "step": 5,
            "name": "Edge Generation",
            "success": True,
            "edges_generated": prereq_result.edge_count
        })
        
        # =================================================================
        # 6단계: 3D 좌표 생성
        # =================================================================
        self.update_state(state='PROGRESS', meta={'step': 6, 'message': '3D 좌표 생성 중...'})
        
        from services.knowledge.visualization import GalaxyVisualizer
        
        vis_nodes = []
        for n in saved_nodes:
            emb = n.get_embedding()
            if emb is not None:
                vis_nodes.append({
                    'id': str(n.id),
                    'title': n.title,
                    'embedding': emb,
                    'cluster_id': n.cluster_id
                })
        
        visualizer = GalaxyVisualizer(scale=100.0)
        viz_result = visualizer.generate_coordinates(nodes=vis_nodes)
        visualizer.save_to_db()
        
        result["steps"].append({
            "step": 6,
            "name": "3D Visualization",
            "success": True,
            "coordinates_generated": viz_result.num_nodes
        })
        
        # =================================================================
        # 완료
        # =================================================================
        result["status"] = "completed"
        result["message"] = f"파이프라인 완료: {result['nodes_created']}개 노드, {result['edges_created']}개 엣지 생성"
        
        return result
        
    except Exception as e:
        logger.exception(f"파이프라인 오류: {e}")
        result["status"] = "failed"
        result["error"] = str(e)
        return result


@shared_task(bind=True, name='knowledge.tasks.regenerate_coordinates')
def regenerate_coordinates(self) -> Dict[str, Any]:
    """
    모든 노드의 3D 좌표 재생성
    """
    try:
        self.update_state(state='PROGRESS', meta={'message': '3D 좌표 재생성 중...'})
        
        from services.knowledge.visualization import GalaxyVisualizer
        
        visualizer = GalaxyVisualizer(scale=100.0)
        result = visualizer.generate_coordinates()
        updated = visualizer.save_to_db()
        
        return {
            "status": "completed",
            "nodes_updated": updated,
            "clusters": result.num_clusters
        }
        
    except Exception as e:
        logger.exception(f"좌표 재생성 오류: {e}")
        return {"status": "failed", "error": str(e)}
