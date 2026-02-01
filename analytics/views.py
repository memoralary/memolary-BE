"""
Cognitive Benchmark API Views

초기 인지 실험(벤치마크) 관련 API 엔드포인트:
- POST /api/benchmark/initialize: 벤치마크 초기화
- POST /api/benchmark/submit: 테스트 결과 제출
- GET /api/benchmark/analyze: 결과 분석
"""

import logging
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.utils import timezone
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

from analytics.models import User, TestSession, TestResult, TestType
from knowledge.models import KnowledgeNode
from services.cognitive.benchmark import (
    CognitiveBenchmark,
    BenchmarkReporter,
    Domain,
)

logger = logging.getLogger(__name__)


class BenchmarkInitializeView(APIView):
    """벤치마크 초기화 API"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="벤치마크 테스트 초기화",
        description="""
새로운 사용자의 인지 벤치마크 테스트 세션을 생성합니다.

### 처리 과정
1. 사용자 생성 또는 조회
2. 도메인별(CS/사투리) 테스트 노드 할당
3. 시간 포인트별(T0~T7) 세션 생성

### 시간 포인트
- T0: 직후
- T1: 20분 후
- T2: 1시간 후
- T3: 9시간 후
- T4: 1일 후
- T5: 2일 후
- T6: 6일 후
- T7: 31일 후
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'username': {'type': 'string', 'description': '사용자 이름 (필수)'},
                    'nodes_per_domain': {'type': 'integer', 'default': 10, 'description': '도메인별 테스트 문항 수'},
                },
                'required': ['username']
            }
        },
        responses={
            201: {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string', 'format': 'uuid'},
                    'username': {'type': 'string'},
                    'created': {'type': 'boolean'},
                    'sessions': {'type': 'object'},
                    'cs_nodes': {'type': 'array'},
                    'dialect_nodes': {'type': 'array'},
                }
            },
            400: {'description': 'username 누락'},
            500: {'description': '서버 오류'},
        }
    )
    def post(self, request):
        try:
            username = request.data.get('username')
            nodes_per_domain = request.data.get('nodes_per_domain', 10)
            
            if not username:
                return Response(
                    {"error": "username is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # 사용자 조회 또는 생성
            user, created = User.objects.get_or_create(
                username=username,
                defaults={"alpha_user": 1.0, "base_forgetting_k": 0.5}
            )
            
            # 벤치마크 초기화
            benchmark = CognitiveBenchmark(nodes_per_domain=nodes_per_domain)
            init_data = benchmark.initialize_benchmark(user)
            
            # 응답 포맷
            sessions_response = {}
            for tp, session in init_data['sessions'].items():
                sessions_response[tp] = {
                    "id": str(session.id),
                    "scheduled_at": session.scheduled_at.isoformat(),
                    "time_point_display": session.get_time_point_display()
                }
            
            cs_nodes = [
                {
                    "id": str(n.id),
                    "title": n.title,
                    "description": n.description,
                    "tags": n.tags
                }
                for n in init_data['nodes'][Domain.CS.value]
            ]
            
            dialect_nodes = [
                {
                    "id": str(n.id),
                    "title": n.title,
                    "description": n.description,
                    "tags": n.tags
                }
                for n in init_data['nodes'][Domain.DIALECT.value]
            ]
            
            return Response({
                "user_id": str(user.id),
                "username": user.username,
                "created": created,
                "sessions": sessions_response,
                "cs_nodes": cs_nodes,
                "dialect_nodes": dialect_nodes,
                "cs_node_ids": init_data['cs_node_ids'],
                "dialect_node_ids": init_data['dialect_node_ids']
            }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Benchmark initialization error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BenchmarkSubmitView(APIView):
    """인지 상태 이벤트 수집 API (테스트 결과 제출)"""
    
    # recall_state → is_correct 변환 매핑
    RECALL_STATE_MAPPING = {
        "REMEMBERED": True,
        "FORGOT": False,
    }
    
    @extend_schema(
        tags=['Analytics'],
        summary="테스트 결과 제출",
        description="""
사용자의 문제 풀이 결과를 수집합니다.

### recall_state 필드 (권장)
- `REMEMBERED`: 기억함 (is_correct = true)
- `FORGOT`: 잊어버림 (is_correct = false)

### test_type
- `A1_BOTTOM_UP`: 상향식 테스트
- `A2_TOP_DOWN`: 하향식 테스트
- `B_RECALL`: 회상 테스트

### 응답
- `illusion_score`: 자신감 - 실제 정확도 (착각 지수)
- `node_stability`: 노드 안정성 점수
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'session_id': {'type': 'string', 'format': 'uuid', 'description': '세션 ID'},
                    'node_id': {'type': 'string', 'format': 'uuid', 'description': '노드 ID'},
                    'recall_state': {'type': 'string', 'enum': ['REMEMBERED', 'FORGOT'], 'description': '기억 상태'},
                    'confidence_score': {'type': 'number', 'minimum': 0, 'maximum': 1, 'description': '확신도'},
                    'response_time_ms': {'type': 'integer', 'description': '반응 시간 (ms)'},
                    'test_type': {'type': 'string', 'enum': ['A1_BOTTOM_UP', 'A2_TOP_DOWN', 'B_RECALL']},
                },
                'required': ['session_id', 'node_id', 'recall_state', 'confidence_score', 'response_time_ms', 'test_type']
            }
        },
        responses={
            201: {
                'type': 'object',
                'properties': {
                    'result_id': {'type': 'integer'},
                    'illusion_score': {'type': 'number'},
                    'node_stability': {'type': 'number'},
                }
            },
            400: {'description': '잘못된 요청'},
            404: {'description': '세션 또는 노드를 찾을 수 없음'},
        }
    )
    def post(self, request):
        try:
            session_id = request.data.get('session_id')
            node_id = request.data.get('node_id')
            confidence_score = request.data.get('confidence_score')
            response_time_ms = request.data.get('response_time_ms')
            test_type = request.data.get('test_type')
            speech_data = request.data.get('speech_data')
            
            # =========================================================
            # recall_state → is_correct 변환 로직
            # =========================================================
            recall_state = request.data.get('recall_state')  # 신규 필드
            is_correct = request.data.get('is_correct')      # 레거시 필드
            
            if recall_state is not None:
                # recall_state가 제공된 경우: enum 유효성 검증 후 변환
                if recall_state not in self.RECALL_STATE_MAPPING:
                    return Response(
                        {
                            "error": f"Invalid recall_state value: '{recall_state}'. "
                                     f"Expected one of: {list(self.RECALL_STATE_MAPPING.keys())}"
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )
                # recall_state를 is_correct로 변환
                is_correct = self.RECALL_STATE_MAPPING[recall_state]
            
            elif is_correct is None:
                # recall_state도 없고 is_correct도 없으면 에러
                return Response(
                    {
                        "error": "Either 'recall_state' or 'is_correct' must be provided. "
                                 "recall_state should be 'REMEMBERED' or 'FORGOT'."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            # else: is_correct만 있는 경우 그대로 사용 (backward compatibility)
            
            # =========================================================
            # 필수 필드 검증 (is_correct는 위에서 처리됨)
            # =========================================================
            required_fields = ['session_id', 'node_id', 
                             'confidence_score', 'response_time_ms', 'test_type']
            missing = [f for f in required_fields if request.data.get(f) is None]
            if missing:
                return Response(
                    {"error": f"Missing required fields: {missing}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # 세션 및 노드 조회
            try:
                session = TestSession.objects.get(id=session_id)
            except TestSession.DoesNotExist:
                return Response(
                    {"error": f"Session not found: {session_id}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            try:
                node = KnowledgeNode.objects.get(id=node_id)
            except KnowledgeNode.DoesNotExist:
                return Response(
                    {"error": f"Node not found: {node_id}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # 결과 기록 (illusion_score는 TestResult.save()에서 자동 계산됨)
            benchmark = CognitiveBenchmark()
            result = benchmark.submit_result(
                session=session,
                node=node,
                is_correct=is_correct,  # recall_state에서 변환된 값 또는 직접 전달된 값
                confidence_score=confidence_score,
                response_time_ms=response_time_ms,
                test_type=test_type,
                speech_data=speech_data
            )
            
            # 세션 performed_at 업데이트
            if session.performed_at is None:
                session.performed_at = timezone.now()
                session.save(update_fields=['performed_at'])
            
            return Response({
                "result_id": result.id,
                "illusion_score": result.illusion_score,
                "node_stability": node.stability_index,
                "node_difficulty": node.difficulty_index
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.exception(f"Result submission error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BenchmarkAnalyzeView(APIView):
    """벤치마크 분석 API"""
    
    # 기본 목표 암기율
    DEFAULT_TARGET_RETENTION = 0.8
    
    @extend_schema(
        tags=['Analytics'],
        summary="벤치마크 결과 분석",
        description="""
벤치마크 테스트 결과를 분석하여 사용자의 인지 특성을 파악합니다.

### 분석 결과
- **망각 계수 (k)**: 에빙하우스 망각곡선 기반 개인화된 망각 속도
- **복습 스케줄**: 목표 암기율 유지를 위한 최적 복습 시점
- **인지 해석**: 학습 유형 및 추천 전략

### 응답 구조
- `summary`: 전체 요약
- `forgetting_curve`: 프론트엔드 시각화용 망각곡선 데이터
- `recommended_review`: 도메인별 복습 시간 (hours, label, curve_x)
- `cognitive_interpretation`: 인지 특성 해석
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string', 'format': 'uuid'},
                    'cs_node_ids': {'type': 'array', 'items': {'type': 'string'}},
                    'dialect_node_ids': {'type': 'array', 'items': {'type': 'string'}},
                    'target_retention': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 0.8},
                },
                'required': ['user_id', 'cs_node_ids', 'dialect_node_ids']
            }
        },
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'summary': {'type': 'object'},
                    'forgetting_curve': {
                        'type': 'object',
                        'properties': {
                            'cs': {'type': 'array'},
                            'dialect': {'type': 'array'},
                        }
                    },
                    'recommended_review': {
                        'type': 'object',
                        'properties': {
                            'target_retention': {'type': 'number'},
                            'cs': {
                                'type': 'object',
                                'properties': {
                                    'hours': {'type': 'number'},
                                    'label': {'type': 'string'},
                                    'curve_x': {'type': 'number'},
                                }
                            },
                            'dialect': {'type': 'object'},
                        }
                    },
                    'cognitive_interpretation': {'type': 'object'},
                }
            },
            400: {'description': '잘못된 요청'},
            404: {'description': '사용자를 찾을 수 없음'},
        }
    )
    def post(self, request):
        try:
            user_id = request.data.get('user_id')
            cs_node_ids = request.data.get('cs_node_ids', [])
            dialect_node_ids = request.data.get('dialect_node_ids', [])
            target_retention = request.data.get('target_retention', self.DEFAULT_TARGET_RETENTION)
            
            if not user_id:
                return Response(
                    {"error": "user_id is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if not cs_node_ids or not dialect_node_ids:
                return Response(
                    {"error": "cs_node_ids and dialect_node_ids are required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # target_retention 유효성 검증
            if not (0 < target_retention < 1):
                return Response(
                    {"error": "target_retention must be between 0 and 1 (exclusive)"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # 분석 실행
            benchmark = CognitiveBenchmark()
            analysis_result = benchmark.analyze_results(
                user_id=user_id,
                cs_node_ids=cs_node_ids,
                dialect_node_ids=dialect_node_ids
            )
            
            # 리포트 생성 (목표 암기율 전달)
            reporter = BenchmarkReporter()
            report = reporter.generate_report(
                analysis_result, 
                target_retention=target_retention
            )
            
            return Response(report, status=status.HTTP_200_OK)
            
        except User.DoesNotExist:
            return Response(
                {"error": f"User not found: {user_id}"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.exception(f"Analysis error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BenchmarkStatusView(APIView):
    """벤치마크 진행 상태 API"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="벤치마크 진행 상태 조회",
        description="""
사용자의 벤치마크 테스트 진행 상황을 조회합니다.

### 응답 구조
- `sessions`: 시간 포인트별 세션 정보
- `completed_results`: 완료된 테스트 결과 수
- `progress_percent`: 전체 진행률 (%)
        """,
        parameters=[
            OpenApiParameter(
                name='user_id',
                type=OpenApiTypes.UUID,
                location=OpenApiParameter.PATH,
                description='사용자 ID'
            )
        ],
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string'},
                    'username': {'type': 'string'},
                    'sessions': {'type': 'array'},
                    'completed_results': {'type': 'integer'},
                    'total_expected': {'type': 'integer'},
                    'progress_percent': {'type': 'number'},
                }
            },
            404: {'description': '사용자를 찾을 수 없음'},
        }
    )
    def get(self, request, user_id):
        try:
            user = User.objects.get(id=user_id)
            
            sessions = TestSession.objects.filter(user=user).order_by('scheduled_at')
            
            session_data = []
            total_results = 0
            
            for session in sessions:
                result_count = TestResult.objects.filter(session=session).count()
                total_results += result_count
                
                session_data.append({
                    "id": str(session.id),
                    "time_point": session.time_point,
                    "time_point_display": session.get_time_point_display(),
                    "scheduled_at": session.scheduled_at.isoformat() if session.scheduled_at else None,
                    "performed_at": session.performed_at.isoformat() if session.performed_at else None,
                    "result_count": result_count,
                    "is_completed": session.performed_at is not None
                })
            
            # 예상 총 결과 수 (4 세션 × 20 노드)
            expected_total = 80
            progress = (total_results / expected_total * 100) if expected_total > 0 else 0
            
            return Response({
                "user_id": str(user.id),
                "username": user.username,
                "sessions": session_data,
                "completed_results": total_results,
                "total_expected": expected_total,
                "progress_percent": round(progress, 1)
            }, status=status.HTTP_200_OK)
            
        except User.DoesNotExist:
            return Response(
                {"error": f"User not found: {user_id}"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.exception(f"Status check error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
