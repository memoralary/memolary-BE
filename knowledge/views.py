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
from django.core.exceptions import ValidationError
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

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
    
    @extend_schema(
        tags=['Knowledge'],
        summary="지식 데이터 수집 (Ingestion)",
        description="""
텍스트 또는 PDF 파일을 업로드하여 지식 그래프를 생성합니다.

### 처리 과정
1. 텍스트/PDF 청킹
2. LLM 기반 노드 추출
3. 클러스터링 및 좌표 할당
4. 선수학습 관계(Edge) 생성

### 비동기 모드 (기본)
- Celery 태스크로 백그라운드 처리
- `task_id`로 진행 상태 조회 가능

### 동기 모드
- `async_mode=false`로 즉시 처리
- 대용량 파일의 경우 타임아웃 주의
        """,
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'text': {'type': 'string', 'description': '처리할 텍스트 내용'},
                    'file': {'type': 'string', 'format': 'binary', 'description': 'PDF 파일'},
                    'async_mode': {'type': 'boolean', 'default': True, 'description': '비동기 처리 여부'},
                },
            }
        },
        responses={
            202: OpenApiExample(
                'Async Response',
                value={
                    'task_id': 'abc123-def456',
                    'status': 'pending',
                    'message': '작업이 시작되었습니다.'
                },
                description='비동기 모드: 태스크 ID 반환'
            ),
            200: OpenApiExample(
                'Sync Response',
                value={
                    'status': 'completed',
                    'nodes_created': 10,
                    'edges_created': 15,
                    'message': '완료: 10개 노드, 15개 엣지'
                },
                description='동기 모드: 처리 결과 즉시 반환'
            ),
            400: {'description': '잘못된 요청 (텍스트/파일 누락)'},
            500: {'description': '서버 오류'},
        }
    )
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
        
        # 파일 처리 - 영구 저장
        file_path = None
        if file:
            ext = os.path.splitext(file.name)[1]
            # 영구 저장 경로 설정
            upload_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'media', 'uploads'
            )
            os.makedirs(upload_dir, exist_ok=True)
            
            # UUID 기반 고유 파일명 생성
            unique_filename = f"{uuid.uuid4()}{ext}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # 파일 저장
            with open(file_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            
            logger.info(f"[PDF_SAVE] 파일 저장 완료: {file_path} ({os.path.getsize(file_path)} bytes)")
        
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
        
        # 3D 좌표 생성을 위해 데이터 준비 (DB 재조회 방지)
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
        visualizer.generate_coordinates(nodes=vis_nodes)
        visualizer.save_to_db()
        
        result["status"] = "completed"
        result["message"] = f"완료: {result['nodes_created']}개 노드, {result['edges_created']}개 엣지"
        
        return result


# =============================================================================
# Task Status API
# =============================================================================

class TaskStatusView(APIView):
    """GET /api/v1/tasks/<task_id>/"""
    
    @extend_schema(
        tags=['Tasks'],
        summary="비동기 작업 상태 조회",
        description="""
Celery 비동기 작업의 현재 상태를 조회합니다.

### 상태 값
- `PENDING`: 대기 중
- `PROGRESS`: 진행 중 (progress 필드에 상세 정보)
- `SUCCESS`: 완료 (result 필드에 결과)
- `FAILURE`: 실패 (error 필드에 오류 메시지)
        """,
        parameters=[
            OpenApiParameter(
                name='task_id',
                type=OpenApiTypes.STR,
                location=OpenApiParameter.PATH,
                description='Celery 태스크 ID'
            )
        ],
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'status': {'type': 'string', 'enum': ['PENDING', 'PROGRESS', 'SUCCESS', 'FAILURE']},
                    'progress': {'type': 'object', 'description': '진행 정보 (PROGRESS 상태일 때)'},
                    'result': {'type': 'object', 'description': '완료 결과 (SUCCESS 상태일 때)'},
                    'error': {'type': 'string', 'description': '오류 메시지 (FAILURE 상태일 때)'},
                }
            }
        }
    )
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
    
    @extend_schema(
        tags=['Universe'],
        summary="3D 시각화 데이터 조회",
        description="""
지식 그래프의 3D 시각화를 위한 전체 노드 및 엣지 데이터를 반환합니다.

### 응답 구조
- **nodes**: 노드 배열 (id, title, position, cluster_id 등)
- **edges**: 엣지 배열 (source, target, relation_type, confidence)
- **metadata**: 통계 정보 (총 노드 수, 엣지 수, 클러스터 목록)

### 좌표 시스템
- x, y, z 좌표는 클러스터 기반 3D 배치
- 동일 클러스터 노드는 인접 배치
        """,
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'nodes': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'id': {'type': 'string'},
                                'title': {'type': 'string'},
                                'description': {'type': 'string'},
                                'cluster_id': {'type': 'integer'},
                                'position': {
                                    'type': 'object',
                                    'properties': {
                                        'x': {'type': 'number'},
                                        'y': {'type': 'number'},
                                        'z': {'type': 'number'},
                                    }
                                }
                            }
                        }
                    },
                    'edges': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'source': {'type': 'string'},
                                'target': {'type': 'string'},
                                'relation_type': {'type': 'string'},
                                'confidence': {'type': 'number'},
                            }
                        }
                    },
                    'metadata': {
                        'type': 'object',
                        'properties': {
                            'total_nodes': {'type': 'integer'},
                            'total_edges': {'type': 'integer'},
                            'total_clusters': {'type': 'integer'},
                        }
                    }
                }
            }
        }
    )
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
    
    @extend_schema(
        tags=['Knowledge'],
        summary="노드 목록 조회",
        description="생성된 지식 노드 목록을 조회합니다 (최대 100개).",
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'count': {'type': 'integer', 'example': 10},
                    'nodes': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'id': {'type': 'string', 'format': 'uuid'},
                                'title': {'type': 'string'},
                                'description': {'type': 'string'},
                                'cluster_id': {'type': 'integer'},
                                'x': {'type': 'number'},
                                'y': {'type': 'number'},
                                'z': {'type': 'number'},
                            }
                        }
                    }
                }
            }
        }
    )
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
    
    @extend_schema(
        tags=['Knowledge'],
        summary="엣지 목록 조회",
        description="노드 간의 관계(엣지) 목록을 조회합니다 (최대 100개).",
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'count': {'type': 'integer', 'example': 15},
                    'edges': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'id': {'type': 'string', 'format': 'uuid'},
                                'source': {'type': 'string'},
                                'target': {'type': 'string'},
                                'source_title': {'type': 'string'},
                                'target_title': {'type': 'string'},
                                'relation_type': {'type': 'string'},
                                'confidence': {'type': 'number'},
                            }
                        }
                    }
                }
            }
        }
    )
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


# =============================================================================
# Recommend API (Edge-based Prerequisite Recommendation)
# =============================================================================

class RecommendView(APIView):
    """
    GET /api/v1/knowledge/recommend/
    
    지식 그래프 기반 추천 API
    """
    
    @extend_schema(
        tags=['Knowledge'],
        summary="다음 학습 추천",
        description="""
특정 노드를 기준으로 다음에 배울 개념을 추천합니다.

### 추천 기준
- 해당 노드에서 나가는 `prerequisite` 관계 사용
- `confidence` 높은 순으로 정렬
        """,
        parameters=[
            OpenApiParameter(
                name='node_id',
                type=OpenApiTypes.UUID,
                location=OpenApiParameter.QUERY,
                description='추천 기준 노드 ID',
                required=True
            ),
            OpenApiParameter(
                name='top_k',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description='추천 개수',
                default=5
            ),
        ],
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'node_id': {'type': 'string'},
                    'top_k': {'type': 'integer'},
                    'recommended': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'id': {'type': 'string'},
                                'title': {'type': 'string'},
                                'confidence': {'type': 'number'},
                            }
                        }
                    }
                }
            },
            400: {'description': 'node_id 누락 또는 잘못된 top_k'},
            404: {'description': '노드를 찾을 수 없음'},
        }
    )
    def get(self, request):
        # =====================================================================
        # Step 1: 파라미터 검증
        # =====================================================================
        node_id = request.query_params.get('node_id')
        
        if not node_id:
            return Response(
                {"error": "node_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        top_k_param = request.query_params.get('top_k', '5')
        
        try:
            top_k = int(top_k_param)
        except ValueError:
            return Response(
                {"error": "top_k must be a valid integer"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # =====================================================================
        # Step 2: 기준 노드 존재 확인
        # =====================================================================
        try:
            source_node = KnowledgeNode.objects.get(id=node_id)
            logger.info(f"[RECOMMEND] 기준 노드: {source_node.title} (id={node_id})")
        except (KnowledgeNode.DoesNotExist, ValidationError, ValueError) as e:
            # ValidationError/ValueError: invalid UUID format
            # DoesNotExist: valid UUID but node doesn't exist
            logger.warning(f"[RECOMMEND] 노드 조회 실패: {node_id} - {type(e).__name__}: {e}")
            return Response(
                {"error": f"invalid or not found node_id: {node_id}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # =====================================================================
        # Step 3: Outgoing Edge 조회 (source_id = node_id)
        # prerequisite 관계는 "다음에 배울 개념"을 의미
        # =====================================================================
        all_outgoing_edges = KnowledgeEdge.objects.filter(source_id=node_id)
        outgoing_count = all_outgoing_edges.count()
        logger.info(f"[RECOMMEND] Outgoing edge 전체: {outgoing_count}건")
        
        # =====================================================================
        # Step 4: relation_type="prerequisite" 필터 적용
        # =====================================================================
        prerequisite_edges = all_outgoing_edges.filter(relation_type="prerequisite")
        filtered_count = prerequisite_edges.count()
        logger.info(f"[RECOMMEND] prerequisite 필터 후: {filtered_count}건")
        
        # =====================================================================
        # Step 5: Self-reference 제외 + confidence 정렬
        # - self-reference: source_id == target_id인 경우 제외
        # - 정렬: confidence DESC, id ASC (안정 정렬)
        # =====================================================================
        edges_with_targets = prerequisite_edges.exclude(
            target_id=node_id  # self-reference 명시적 제외
        ).select_related('target').order_by('-confidence', 'id')
        
        # =====================================================================
        # Step 6: 추천 결과 생성
        # =====================================================================
        recommended = []
        for edge in edges_with_targets[:top_k]:
            target_node = edge.target
            confidence_val = float(edge.confidence) if edge.confidence is not None else 1.0
            
            recommended.append({
                "id": str(target_node.id),
                "title": target_node.title,
                "confidence": confidence_val,
            })
            
            # 디버깅: 각 추천 항목 로그
            logger.info(
                f"[RECOMMEND] 추천: {target_node.title} "
                f"(id={target_node.id}, confidence={confidence_val})"
            )
        
        # =====================================================================
        # Step 7: 결과가 0개인 경우 원인 로그
        # =====================================================================
        if len(recommended) == 0:
            if outgoing_count == 0:
                logger.warning(f"[RECOMMEND] 추천 0건: outgoing edge 없음 (node_id={node_id})")
            elif filtered_count == 0:
                logger.warning(
                    f"[RECOMMEND] 추천 0건: prerequisite 관계 없음 "
                    f"(outgoing={outgoing_count}건, 모두 다른 relation_type)"
                )
            else:
                logger.warning(
                    f"[RECOMMEND] 추천 0건: self-reference만 존재 "
                    f"(prerequisite={filtered_count}건)"
                )
        
        logger.info(f"[RECOMMEND] 최종 추천: {[r['id'] for r in recommended]}")
        
        return Response({
            "node_id": node_id,
            "top_k": top_k,
            "recommended": recommended
        })
