"""
Knowledge API Views

엔드포인트:
- POST /api/v1/knowledge/ingest/ - 텍스트/PDF 파이프라인 실행
- GET /api/v1/universe/ - 은하수 시각화 데이터
- GET /api/v1/tasks/<task_id>/ - 태스크 상태 조회
"""

import os
import uuid
import tempfile
import logging
from typing import Optional

from django.db import connection
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from knowledge.models import KnowledgeNode, KnowledgeEdge
from knowledge.serializers import IngestionRequestSerializer

logger = logging.getLogger(__name__)


# =============================================================================
# Ingestion API
# =============================================================================

class IngestionView(APIView):
    """
    POST /api/v1/knowledge/ingest/
    
    텍스트 또는 PDF를 받아 Knowledge Graph 파이프라인 실행
    """
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    
    def post(self, request):
        serializer = IngestionRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        text = serializer.validated_data.get('text')
        file = serializer.validated_data.get('file')
        async_mode = serializer.validated_data.get('async_mode', True)
        
        # 파일 처리
        file_path = None
        if file:
            ext = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=ext,
                dir='/tmp'
            ) as tmp:
                for chunk in file.chunks():
                    tmp.write(chunk)
                file_path = tmp.name
        
        if async_mode:
            try:
                from knowledge.tasks import process_ingestion
                
                task = process_ingestion.delay(
                    text=text,
                    file_path=file_path
                )
                
                return Response({
                    "task_id": task.id,
                    "status": "pending",
                    "message": "작업이 시작되었습니다."
                }, status=status.HTTP_202_ACCEPTED)
                
            except Exception as e:
                logger.warning(f"Celery 연결 실패, 동기 모드로 전환: {e}")
                async_mode = False
        
        if not async_mode:
            try:
                result = self._process_sync(text, file_path)
                return Response(result, status=status.HTTP_200_OK)
            except Exception as e:
                logger.exception(f"처리 오류: {e}")
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
    
    def _process_sync(self, text: Optional[str], file_path: Optional[str]) -> dict:
        """동기 처리"""
        from dotenv import load_dotenv
        load_dotenv()
        
        from services.knowledge.ingestion import IngestionService
        from services.knowledge.extractor import extract_nodes
        from services.knowledge.clustering import ClusteringService
        from services.knowledge.curriculum import generate_prerequisites
        from services.knowledge.visualization import GalaxyVisualizer
        
        result = {"status": "processing", "nodes_created": 0, "edges_created": 0}
        
        service = IngestionService(chunk_size=2000)
        if text:
            ingestion_result = service.process(text)
        elif file_path:
            ingestion_result = service.process(file_path)
        else:
            raise ValueError("텍스트 또는 파일이 필요합니다.")
        
        existing_titles = list(KnowledgeNode.objects.values_list('title', flat=True))
        all_nodes = []
        
        for chunk in ingestion_result.chunks:
            try:
                extraction_result = extract_nodes(chunk, existing_titles)
                all_nodes.extend(extraction_result.nodes)
                existing_titles.extend([n.title for n in extraction_result.nodes])
            except Exception as e:
                logger.warning(f"청크 추출 실패: {e}")
        
        if not all_nodes:
            result["status"] = "completed"
            result["message"] = "추출된 노드가 없습니다."
            return result
        
        clustering_service = ClusteringService(
            model_name='paraphrase-MiniLM-L6-v2',
            similarity_threshold=0.7
        )
        
        nodes_for_clustering = [
            {"title": node.title, "description": node.description or ""}
            for node in all_nodes
        ]
        cluster_results = clustering_service.assign_clusters_batch(nodes_for_clustering)
        
        saved_nodes = []
        for node, cluster in zip(all_nodes, cluster_results):
            existing = KnowledgeNode.objects.filter(title=node.title).first()
            
            if existing:
                saved_nodes.append(existing)
            else:
                new_node = KnowledgeNode.objects.create(
                    title=node.title,
                    description=node.description or "",
                    cluster_id=cluster.cluster_id,
                    tags=node.tags,
                )
                new_node.set_embedding(cluster.embedding)
                new_node.save()
                saved_nodes.append(new_node)
                result["nodes_created"] += 1
        
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
                _, created = KnowledgeEdge.objects.get_or_create(
                    source=source_node,
                    target=target_node,
                    relation_type="prerequisite",
                    defaults={"confidence": edge.confidence, "is_prerequisite": True}
                )
                if created:
                    result["edges_created"] += 1
        
        visualizer = GalaxyVisualizer(scale=100.0)
        visualizer.generate_coordinates()
        visualizer.save_to_db()
        
        result["status"] = "completed"
        result["message"] = f"완료: {result['nodes_created']}개 노드, {result['edges_created']}개 엣지"
        
        return result


# =============================================================================
# Task Status API
# =============================================================================

class TaskStatusView(APIView):
    """GET /api/v1/tasks/<task_id>/"""
    
    def get(self, request, task_id):
        try:
            from celery.result import AsyncResult
            from backend.celery import app
            
            result = AsyncResult(task_id, app=app)
            
            response = {"task_id": task_id, "status": result.status}
            
            if result.status == 'PROGRESS':
                response["progress"] = result.info
            elif result.status == 'SUCCESS':
                response["result"] = result.result
            elif result.status == 'FAILURE':
                response["error"] = str(result.result)
            
            return Response(response, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# =============================================================================
# Universe API
# =============================================================================

class UniverseView(APIView):
    """GET /api/v1/universe/ - 은하수 시각화 데이터"""
    
    def get(self, request):
        nodes_data = []
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, description, cluster_id, x, y, z, tags
                FROM knowledge_knowledgenode
                ORDER BY created_at DESC
            """)
            columns = [col[0] for col in cursor.description]
            
            for row in cursor.fetchall():
                node = dict(zip(columns, row))
                nodes_data.append({
                    "id": str(node['id']),
                    "title": node['title'],
                    "description": node['description'] or "",
                    "cluster_id": node['cluster_id'],
                    "position": {
                        "x": float(node['x']) if node['x'] else 0,
                        "y": float(node['y']) if node['y'] else 0,
                        "z": float(node['z']) if node['z'] else 0,
                    },
                    "tags": node['tags'] if node['tags'] else [],
                })
        
        edges_data = []
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT source_id, target_id, relation_type, confidence
                FROM knowledge_knowledgeedge
            """)
            
            for row in cursor.fetchall():
                edges_data.append({
                    "source": str(row[0]),
                    "target": str(row[1]),
                    "relation_type": row[2],
                    "confidence": float(row[3]) if row[3] else 1.0,
                })
        
        clusters = list(set(n['cluster_id'] for n in nodes_data if n['cluster_id']))
        
        return Response({
            "nodes": nodes_data,
            "edges": edges_data,
            "metadata": {
                "total_nodes": len(nodes_data),
                "total_edges": len(edges_data),
                "total_clusters": len(clusters),
                "clusters": clusters,
            }
        })


# =============================================================================
# CRUD API
# =============================================================================

class NodeListView(APIView):
    """GET /api/v1/knowledge/nodes/"""
    
    def get(self, request):
        nodes = KnowledgeNode.objects.all()[:100]
        
        data = [{
            "id": str(n.id),
            "title": n.title,
            "description": n.description,
            "cluster_id": n.cluster_id,
            "x": n.x, "y": n.y, "z": n.z,
        } for n in nodes]
        
        return Response({"count": len(data), "nodes": data})


class EdgeListView(APIView):
    """GET /api/v1/knowledge/edges/"""
    
    def get(self, request):
        edges = KnowledgeEdge.objects.select_related('source', 'target').all()[:100]
        
        data = [{
            "id": str(e.id),
            "source": str(e.source_id),
            "target": str(e.target_id),
            "source_title": e.source.title,
            "target_title": e.target.title,
            "relation_type": e.relation_type,
            "confidence": e.confidence,
        } for e in edges]
        
        return Response({"count": len(data), "edges": data})
