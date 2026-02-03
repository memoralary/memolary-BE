"""
Celery 태스크 - Knowledge Graph Pipeline

백그라운드에서 실행되는 무거운 작업들
"""

import os
import logging
from typing import Dict, Any, Optional
from celery import shared_task
from celery.exceptions import Ignore

logger = logging.getLogger(__name__)


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
        
        service = IngestionService(chunk_size=2000)
        
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
        # 2단계: 노드 추출
        # =================================================================
        self.update_state(state='PROGRESS', meta={'step': 2, 'message': '노드 추출 중...'})
        
        from services.knowledge.extractor import extract_nodes
        from knowledge.models import KnowledgeNode
        
        existing_titles = list(KnowledgeNode.objects.values_list('title', flat=True))
        
        all_nodes = []
        for chunk in ingestion_result.chunks:
            try:
                extraction_result = extract_nodes(chunk, existing_titles)
                all_nodes.extend(extraction_result.nodes)
                existing_titles.extend([n.title for n in extraction_result.nodes])
            except Exception as e:
                logger.warning(f"청크 추출 실패: {e}")
                result["errors"].append(str(e))
        
        result["steps"].append({
            "step": 2,
            "name": "Node Extraction",
            "success": True,
            "nodes_extracted": len(all_nodes)
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
            {"title": node.title, "description": node.description or ""}
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
        
        saved_nodes = []
        for node, cluster in zip(all_nodes, cluster_results):
            existing = KnowledgeNode.objects.filter(title=node.title).first()
            
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
                    title=node.title,
                    description=node.description or "",
                    cluster_id=cluster.cluster_id,
                    tags=node.tags,
                    track_type=track_type,  # 자동 분류된 track_type 사용
                )
                new_node.set_embedding(cluster.embedding)
                new_node.save()
                saved_nodes.append(new_node)
                result["nodes_created"] += 1
        
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
