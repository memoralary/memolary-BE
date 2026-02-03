"""
Debug API Views - 개발 및 테스트용 디버깅 API

⚠️ 주의: 이 API들은 개발 환경에서만 사용해야 합니다.
프로덕션에서는 접근을 제한하거나 비활성화해야 합니다.

엔드포인트:
- GET  /api/v1/debug/health/ - 시스템 상태 확인
- GET  /api/v1/debug/stats/ - DB 통계
- GET  /api/v1/debug/data/<model_name>/ - 모델 데이터 조회
- POST /api/v1/debug/seed/ - 테스트 데이터 삽입
- POST /api/v1/debug/clear/ - DB 데이터 삭제
- POST /api/v1/debug/reset/ - 전체 초기화 (clear + seed)
"""

import logging
import random
from datetime import timedelta
from typing import Dict, Any, List, Optional

from django.conf import settings
from django.db import connection
from django.utils import timezone
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

logger = logging.getLogger(__name__)


# =============================================================================
# Health Check API
# =============================================================================

class HealthCheckView(APIView):
    """시스템 상태 확인 API"""
    
    @extend_schema(
        tags=['Debug'],
        summary="시스템 상태 확인",
        description="""
개발 환경 시스템 상태를 확인합니다.

### 확인 항목
- 데이터베이스 연결
- Celery 워커 상태
- 환경 변수 설정
        """,
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'database': {'type': 'object'},
                    'celery': {'type': 'object'},
                    'environment': {'type': 'object'},
                }
            }
        }
    )
    def get(self, request):
        result = {
            "status": "ok",
            "timestamp": timezone.now().isoformat(),
        }
        
        # DB 연결 확인
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            result["database"] = {
                "status": "connected",
                "engine": settings.DATABASES['default']['ENGINE'],
            }
        except Exception as e:
            result["database"] = {"status": "error", "error": str(e)}
            result["status"] = "degraded"
        
        # Celery 상태 확인
        try:
            from backend.celery import app
            inspect = app.control.inspect()
            active = inspect.active()
            result["celery"] = {
                "status": "connected" if active else "no_workers",
                "workers": list(active.keys()) if active else [],
            }
        except Exception as e:
            result["celery"] = {"status": "unavailable", "error": str(e)}
        
        # 환경 설정
        result["environment"] = {
            "debug": settings.DEBUG,
            "allowed_hosts": settings.ALLOWED_HOSTS[:3],  # 일부만 표시
        }
        
        return Response(result, status=status.HTTP_200_OK)


# =============================================================================
# Stats API
# =============================================================================

class StatsView(APIView):
    """DB 통계 API"""
    
    @extend_schema(
        tags=['Debug'],
        summary="DB 통계 조회",
        description="모든 테이블의 레코드 수와 기본 통계를 조회합니다.",
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'knowledge': {'type': 'object'},
                    'analytics': {'type': 'object'},
                    'total_records': {'type': 'integer'},
                }
            }
        }
    )
    def get(self, request):
        from knowledge.models import KnowledgeNode, KnowledgeEdge, TrackType
        from analytics.models import User, TestSession, TestResult
        
        # Knowledge 모델 통계
        node_count = KnowledgeNode.objects.count()
        track_a_count = KnowledgeNode.objects.filter(track_type=TrackType.TRACK_A).count()
        track_b_count = KnowledgeNode.objects.filter(track_type=TrackType.TRACK_B).count()
        edge_count = KnowledgeEdge.objects.count()
        
        # Analytics 모델 통계
        user_count = User.objects.count()
        session_count = TestSession.objects.count()
        result_count = TestResult.objects.count()
        
        # 클러스터 통계
        clusters = list(KnowledgeNode.objects.values_list('cluster_id', flat=True).distinct())
        
        return Response({
            "knowledge": {
                "nodes": {
                    "total": node_count,
                    "track_a_cs": track_a_count,
                    "track_b_dialect": track_b_count,
                    "with_coordinates": KnowledgeNode.objects.filter(x__isnull=False).count(),
                    "with_embedding": KnowledgeNode.objects.filter(embedding__isnull=False).count(),
                },
                "edges": edge_count,
                "clusters": len([c for c in clusters if c]),
            },
            "analytics": {
                "users": user_count,
                "sessions": session_count,
                "test_results": result_count,
            },
            "total_records": node_count + edge_count + user_count + session_count + result_count,
            "timestamp": timezone.now().isoformat(),
        }, status=status.HTTP_200_OK)


# =============================================================================
# Data View API
# =============================================================================

class DataView(APIView):
    """모델 데이터 조회 API"""
    
    MODEL_MAPPING = {
        'nodes': 'knowledge.KnowledgeNode',
        'edges': 'knowledge.KnowledgeEdge',
        'users': 'analytics.User',
        'sessions': 'analytics.TestSession',
        'results': 'analytics.TestResult',
    }
    
    @extend_schema(
        tags=['Debug'],
        summary="모델 데이터 조회",
        description="""
특정 모델의 데이터를 조회합니다.

### 지원 모델
- `nodes` - KnowledgeNode
- `edges` - KnowledgeEdge
- `users` - User
- `sessions` - TestSession
- `results` - TestResult
        """,
        parameters=[
            OpenApiParameter(
                name='model_name',
                type=OpenApiTypes.STR,
                location=OpenApiParameter.PATH,
                description='모델 이름',
                enum=['nodes', 'edges', 'users', 'sessions', 'results']
            ),
            OpenApiParameter(
                name='limit',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description='최대 레코드 수',
                default=50
            ),
            OpenApiParameter(
                name='offset',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description='시작 위치',
                default=0
            ),
        ],
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'model': {'type': 'string'},
                    'count': {'type': 'integer'},
                    'data': {'type': 'array'},
                }
            }
        }
    )
    def get(self, request, model_name):
        if model_name not in self.MODEL_MAPPING:
            return Response(
                {"error": f"Unknown model: {model_name}. Available: {list(self.MODEL_MAPPING.keys())}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        limit = int(request.query_params.get('limit', 50))
        offset = int(request.query_params.get('offset', 0))
        
        # 모델 가져오기
        if model_name == 'nodes':
            from knowledge.models import KnowledgeNode
            queryset = KnowledgeNode.objects.all().order_by('-created_at')
            data = [
                {
                    "id": str(n.id),
                    "title": n.title,
                    "description": n.description[:100] if n.description else "",
                    "track_type": n.track_type,
                    "cluster_id": n.cluster_id,
                    "tags": n.tags,
                    "x": n.x, "y": n.y, "z": n.z,
                    "has_embedding": n.embedding is not None,
                    "created_at": n.created_at.isoformat(),
                }
                for n in queryset[offset:offset+limit]
            ]
        
        elif model_name == 'edges':
            from knowledge.models import KnowledgeEdge
            queryset = KnowledgeEdge.objects.select_related('source', 'target').all()
            data = [
                {
                    "id": e.id,
                    "source": str(e.source_id),
                    "source_title": e.source.title,
                    "target": str(e.target_id),
                    "target_title": e.target.title,
                    "relation_type": e.relation_type,
                    "confidence": e.confidence,
                }
                for e in queryset[offset:offset+limit]
            ]
        
        elif model_name == 'users':
            from analytics.models import User
            queryset = User.objects.all().order_by('-created_at')
            data = [
                {
                    "id": str(u.id),
                    "username": u.username,
                    "alpha_user": u.alpha_user,
                    "base_forgetting_k": u.base_forgetting_k,
                    "created_at": u.created_at.isoformat(),
                }
                for u in queryset[offset:offset+limit]
            ]
        
        elif model_name == 'sessions':
            from analytics.models import TestSession
            queryset = TestSession.objects.select_related('user').all().order_by('-scheduled_at')
            data = [
                {
                    "id": str(s.id),
                    "user": s.user.username,
                    "time_point": s.time_point,
                    "scheduled_at": s.scheduled_at.isoformat() if s.scheduled_at else None,
                    "performed_at": s.performed_at.isoformat() if s.performed_at else None,
                }
                for s in queryset[offset:offset+limit]
            ]
        
        elif model_name == 'results':
            from analytics.models import TestResult
            queryset = TestResult.objects.select_related('session', 'node').all().order_by('-created_at')
            data = [
                {
                    "id": r.id,
                    "session_id": str(r.session_id),
                    "node_title": r.node.title,
                    "is_correct": r.is_correct,
                    "confidence_score": r.confidence_score,
                    "response_time_ms": r.response_time_ms,
                    "illusion_score": r.illusion_score,
                    "test_type": r.test_type,
                }
                for r in queryset[offset:offset+limit]
            ]
        
        return Response({
            "model": model_name,
            "total_count": queryset.count(),
            "offset": offset,
            "limit": limit,
            "data": data,
        }, status=status.HTTP_200_OK)


# =============================================================================
# Seed Data API
# =============================================================================

class SeedDataView(APIView):
    """테스트 데이터 삽입 API"""
    
    # 샘플 CS 노드 데이터
    CS_SAMPLE_NODES = [
        {"title": "SQL 기초", "description": "SQL 쿼리의 기본 구조와 문법", "tags": ["데이터베이스", "SQL", "쿼리"]},
        {"title": "인덱스 최적화", "description": "데이터베이스 인덱스를 활용한 성능 최적화", "tags": ["데이터베이스", "인덱스", "성능"]},
        {"title": "정규화", "description": "데이터베이스 정규화 1NF, 2NF, 3NF", "tags": ["데이터베이스", "정규화", "설계"]},
        {"title": "이진 탐색", "description": "정렬된 배열에서 효율적인 탐색", "tags": ["알고리즘", "탐색", "자료구조"]},
        {"title": "해시 테이블", "description": "해시 함수를 이용한 O(1) 탐색", "tags": ["자료구조", "해시", "알고리즘"]},
        {"title": "BFS/DFS", "description": "그래프 탐색 알고리즘", "tags": ["알고리즘", "그래프", "탐색"]},
        {"title": "프로세스 스케줄링", "description": "CPU 스케줄링 알고리즘", "tags": ["운영체제", "프로세스", "스케줄링"]},
        {"title": "TCP/IP", "description": "네트워크 프로토콜 스택", "tags": ["네트워크", "프로토콜", "TCP"]},
        {"title": "REST API", "description": "RESTful 웹 서비스 설계 원칙", "tags": ["API", "웹", "설계"]},
        {"title": "Git 브랜치", "description": "Git 버전 관리와 브랜치 전략", "tags": ["Git", "버전관리", "협업"]},
    ]
    
    # 샘플 Dialect 노드 데이터
    DIALECT_SAMPLE_NODES = [
        {"title": "경상도 억양", "description": "경상도 방언의 억양 특성", "tags": ["경상도", "방언", "억양"]},
        {"title": "전라도 어휘", "description": "전라도 지역 특유의 어휘", "tags": ["전라도", "방언", "어휘"]},
        {"title": "사투리 음운 변화", "description": "지역별 음운 변화 패턴", "tags": ["방언", "음운", "언어학"]},
        {"title": "경상도 존대법", "description": "경상도 방언의 존대 표현", "tags": ["경상도", "방언", "존대법"]},
        {"title": "전라도 관용 표현", "description": "전라도 지역 관용 표현", "tags": ["전라도", "방언", "관용표현"]},
        {"title": "지역 음식 명칭", "description": "지역별 음식 이름 차이", "tags": ["방언", "음식", "지역문화"]},
        {"title": "택호 문화", "description": "지역별 호칭 문화", "tags": ["전라도", "호칭", "민속"]},
        {"title": "민속 어휘", "description": "전통 민속 관련 지역 어휘", "tags": ["민속", "방언", "전통문화"]},
        {"title": "경북 사투리", "description": "경북 지역 사투리 특성", "tags": ["경상도", "경북", "사투리"]},
        {"title": "호남 방언", "description": "호남 지역 방언 특성", "tags": ["전라도", "호남", "방언"]},
    ]
    
    @extend_schema(
        tags=['Debug'],
        summary="테스트 데이터 삽입",
        description="""
테스트용 샘플 데이터를 DB에 삽입합니다.

### 생성 데이터
- CS 노드 10개 (TRACK_A)
- Dialect 노드 10개 (TRACK_B)
- 선수학습 관계 (Edge)
- 테스트 유저 및 세션 (선택)
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'include_users': {'type': 'boolean', 'default': True, 'description': '테스트 유저 생성 여부'},
                    'include_edges': {'type': 'boolean', 'default': True, 'description': '엣지 생성 여부'},
                    'user_count': {'type': 'integer', 'default': 3, 'description': '생성할 유저 수'},
                },
            }
        },
        responses={
            201: {
                'type': 'object',
                'properties': {
                    'nodes_created': {'type': 'integer'},
                    'edges_created': {'type': 'integer'},
                    'users_created': {'type': 'integer'},
                }
            }
        }
    )
    def post(self, request):
        from knowledge.models import KnowledgeNode, KnowledgeEdge, TrackType, RelationType
        from analytics.models import User, TestSession
        
        include_users = request.data.get('include_users', True)
        include_edges = request.data.get('include_edges', True)
        user_count = request.data.get('user_count', 3)
        
        result = {
            "nodes_created": 0,
            "edges_created": 0,
            "users_created": 0,
            "sessions_created": 0,
        }
        
        created_nodes = []
        
        # CS 노드 생성
        for node_data in self.CS_SAMPLE_NODES:
            node, created = KnowledgeNode.objects.get_or_create(
                title=node_data["title"],
                defaults={
                    "description": node_data["description"],
                    "tags": node_data["tags"],
                    "track_type": TrackType.TRACK_A,
                }
            )
            if created:
                result["nodes_created"] += 1
                created_nodes.append(node)
        
        # Dialect 노드 생성
        for node_data in self.DIALECT_SAMPLE_NODES:
            node, created = KnowledgeNode.objects.get_or_create(
                title=node_data["title"],
                defaults={
                    "description": node_data["description"],
                    "tags": node_data["tags"],
                    "track_type": TrackType.TRACK_B,
                }
            )
            if created:
                result["nodes_created"] += 1
                created_nodes.append(node)
        
        # 엣지 생성 (간단한 체인 구조)
        if include_edges and len(created_nodes) >= 2:
            for i in range(len(created_nodes) - 1):
                _, created = KnowledgeEdge.objects.get_or_create(
                    source=created_nodes[i],
                    target=created_nodes[i + 1],
                    relation_type=RelationType.PREREQUISITE,
                    defaults={"confidence": 0.8, "is_prerequisite": True}
                )
                if created:
                    result["edges_created"] += 1
        
        # 테스트 유저 생성
        if include_users:
            for i in range(user_count):
                user, created = User.objects.get_or_create(
                    username=f"test_user_{i+1}",
                    defaults={
                        "alpha_user": random.uniform(0.8, 1.2),
                        "base_forgetting_k": random.uniform(0.3, 0.7),
                    }
                )
                if created:
                    result["users_created"] += 1
                    
                    # 테스트 세션 생성
                    now = timezone.now()
                    for tp, minutes in [('T0', 0), ('T1', 10), ('T2', 60), ('T3', 1440)]:
                        TestSession.objects.get_or_create(
                            user=user,
                            time_point=tp,
                            defaults={"scheduled_at": now + timedelta(minutes=minutes)}
                        )
                        result["sessions_created"] += 1
        
        logger.info(f"[SeedData] 생성 완료: {result}")
        
        return Response({
            **result,
            "message": f"테스트 데이터 생성 완료",
        }, status=status.HTTP_201_CREATED)


# =============================================================================
# Clear Data API
# =============================================================================

class ClearDataView(APIView):
    """DB 데이터 삭제 API"""
    
    @extend_schema(
        tags=['Debug'],
        summary="DB 데이터 삭제",
        description="""
⚠️ **주의**: 이 API는 데이터를 삭제합니다.

### 삭제 대상
- `all`: 모든 데이터 삭제
- `nodes`: KnowledgeNode + KnowledgeEdge
- `analytics`: User + TestSession + TestResult
- `results`: TestResult만
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'target': {
                        'type': 'string',
                        'enum': ['all', 'nodes', 'analytics', 'results'],
                        'default': 'all',
                        'description': '삭제 대상'
                    },
                    'confirm': {
                        'type': 'boolean',
                        'default': False,
                        'description': '삭제 확인 (true여야 실행)'
                    },
                },
            }
        },
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'deleted': {'type': 'object'},
                    'message': {'type': 'string'},
                }
            }
        }
    )
    def post(self, request):
        target = request.data.get('target', 'all')
        confirm = request.data.get('confirm', False)
        
        if not confirm:
            return Response(
                {"error": "삭제를 실행하려면 confirm: true를 전송하세요."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        from knowledge.models import KnowledgeNode, KnowledgeEdge
        from analytics.models import User, TestSession, TestResult
        
        deleted = {}
        
        if target in ['all', 'results']:
            count = TestResult.objects.count()
            TestResult.objects.all().delete()
            deleted['test_results'] = count
        
        if target in ['all', 'analytics']:
            count = TestSession.objects.count()
            TestSession.objects.all().delete()
            deleted['sessions'] = count
            
            count = User.objects.count()
            User.objects.all().delete()
            deleted['users'] = count
        
        if target in ['all', 'nodes']:
            count = KnowledgeEdge.objects.count()
            KnowledgeEdge.objects.all().delete()
            deleted['edges'] = count
            
            count = KnowledgeNode.objects.count()
            KnowledgeNode.objects.all().delete()
            deleted['nodes'] = count
        
        logger.warning(f"[ClearData] 데이터 삭제: {deleted}")
        
        return Response({
            "deleted": deleted,
            "message": f"데이터 삭제 완료 (target={target})",
        }, status=status.HTTP_200_OK)


# =============================================================================
# Reset API (Clear + Seed)
# =============================================================================

class ResetView(APIView):
    """전체 초기화 API (Clear + Seed)"""
    
    @extend_schema(
        tags=['Debug'],
        summary="전체 초기화 (Clear + Seed)",
        description="""
⚠️ **주의**: 모든 데이터를 삭제하고 샘플 데이터로 초기화합니다.

### 처리 순서
1. 모든 데이터 삭제
2. 테스트 노드 생성 (CS 10개, Dialect 10개)
3. 테스트 유저 및 세션 생성
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'confirm': {
                        'type': 'boolean',
                        'default': False,
                        'description': '초기화 확인 (true여야 실행)'
                    },
                },
            }
        },
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'cleared': {'type': 'object'},
                    'seeded': {'type': 'object'},
                }
            }
        }
    )
    def post(self, request):
        confirm = request.data.get('confirm', False)
        
        if not confirm:
            return Response(
                {"error": "초기화를 실행하려면 confirm: true를 전송하세요."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Clear
        clear_view = ClearDataView()
        clear_request = request._request
        clear_request.data = {'target': 'all', 'confirm': True}
        
        from knowledge.models import KnowledgeNode, KnowledgeEdge
        from analytics.models import User, TestSession, TestResult
        
        cleared = {
            'test_results': TestResult.objects.count(),
            'sessions': TestSession.objects.count(),
            'users': User.objects.count(),
            'edges': KnowledgeEdge.objects.count(),
            'nodes': KnowledgeNode.objects.count(),
        }
        
        TestResult.objects.all().delete()
        TestSession.objects.all().delete()
        User.objects.all().delete()
        KnowledgeEdge.objects.all().delete()
        KnowledgeNode.objects.all().delete()
        
        # Seed
        seed_view = SeedDataView()
        seed_request = request
        seed_request.data = {'include_users': True, 'include_edges': True, 'user_count': 3}
        seed_response = seed_view.post(seed_request)
        
        logger.warning(f"[Reset] 전체 초기화 완료")
        
        return Response({
            "cleared": cleared,
            "seeded": seed_response.data,
            "message": "전체 초기화 완료",
        }, status=status.HTTP_200_OK)


# =============================================================================
# Benchmark Quick Test API
# =============================================================================

class QuickBenchmarkView(APIView):
    """벤치마크 빠른 테스트 API"""
    
    @extend_schema(
        tags=['Debug'],
        summary="벤치마크 빠른 테스트",
        description="""
벤치마크 테스트 결과를 빠르게 생성합니다.

### 생성 데이터
- 테스트 유저 생성 (또는 기존 유저 사용)
- T0~T3 세션 및 결과 자동 생성
- 무작위 정답/오답 및 확신도 생성
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'username': {'type': 'string', 'default': 'quick_test_user'},
                    'nodes_per_domain': {'type': 'integer', 'default': 5},
                },
            }
        },
        responses={
            201: {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string'},
                    'sessions_created': {'type': 'integer'},
                    'results_created': {'type': 'integer'},
                }
            }
        }
    )
    def post(self, request):
        from knowledge.models import KnowledgeNode, TrackType
        from analytics.models import User, TestSession, TestResult, TestType
        
        username = request.data.get('username', 'quick_test_user')
        nodes_per_domain = request.data.get('nodes_per_domain', 5)
        
        # 유저 생성
        user, _ = User.objects.get_or_create(
            username=username,
            defaults={"alpha_user": 1.0, "base_forgetting_k": 0.5}
        )
        
        # 노드 선택
        cs_nodes = list(KnowledgeNode.objects.filter(track_type=TrackType.TRACK_A)[:nodes_per_domain])
        dialect_nodes = list(KnowledgeNode.objects.filter(track_type=TrackType.TRACK_B)[:nodes_per_domain])
        
        # 노드가 부족하면 자동으로 seed 실행
        if len(cs_nodes) < nodes_per_domain or len(dialect_nodes) < nodes_per_domain:
            logger.info("[QuickBenchmark] 노드 부족 - 자동 seed 실행")
            seed_view = SeedDataView()
            seed_view.post(request)
            
            # 노드 다시 조회
            cs_nodes = list(KnowledgeNode.objects.filter(track_type=TrackType.TRACK_A)[:nodes_per_domain])
            dialect_nodes = list(KnowledgeNode.objects.filter(track_type=TrackType.TRACK_B)[:nodes_per_domain])
        
        all_nodes = cs_nodes + dialect_nodes
        
        result = {
            "user_id": str(user.id),
            "username": username,
            "sessions_created": 0,
            "results_created": 0,
        }
        
        # 세션 및 결과 생성
        now = timezone.now()
        time_intervals = {'T0': 0, 'T1': 10, 'T2': 60, 'T3': 1440}
        
        # T0 ~ T3 정답률 점차 감소 시뮬레이션
        accuracy_by_timepoint = {'T0': 0.95, 'T1': 0.85, 'T2': 0.70, 'T3': 0.50}
        
        for tp, minutes in time_intervals.items():
            session, created = TestSession.objects.get_or_create(
                user=user,
                time_point=tp,
                defaults={
                    "scheduled_at": now + timedelta(minutes=minutes),
                    "performed_at": now + timedelta(minutes=minutes + 5),
                }
            )
            
            if created:
                result["sessions_created"] += 1
            
            # 각 노드에 대해 결과 생성
            accuracy = accuracy_by_timepoint[tp]
            for node in all_nodes:
                is_correct = random.random() < accuracy
                confidence = random.uniform(0.5, 1.0) if is_correct else random.uniform(0.2, 0.7)
                
                TestResult.objects.get_or_create(
                    session=session,
                    node=node,
                    defaults={
                        "is_correct": is_correct,
                        "confidence_score": round(confidence, 2),
                        "response_time_ms": random.randint(1000, 5000),
                        "test_type": TestType.B_RECALL,
                    }
                )
                result["results_created"] += 1
        
        result["cs_node_ids"] = [str(n.id) for n in cs_nodes]
        result["dialect_node_ids"] = [str(n.id) for n in dialect_nodes]
        result["message"] = f"벤치마크 테스트 데이터 생성 완료"
        
        return Response(result, status=status.HTTP_201_CREATED)
