"""
Review Scheduling API Views - 복습 스케줄 관리 API

엔드포인트:
- GET  /api/v1/analytics/schedules/ - 복습 스케줄 목록 조회
- POST /api/v1/analytics/schedules/ - 복습 스케줄 생성
- POST /api/v1/analytics/schedules/auto/ - 분석 결과 기반 자동 스케줄 생성
- GET  /api/v1/analytics/schedules/<id>/ - 스케줄 상세 조회
- PATCH /api/v1/analytics/schedules/<id>/ - 스케줄 수정
- DELETE /api/v1/analytics/schedules/<id>/ - 스케줄 삭제
- POST /api/v1/analytics/schedules/<id>/complete/ - 복습 완료 표시
- POST /api/v1/analytics/schedules/check-notify/ - 알림 확인 및 전송
- GET  /api/v1/analytics/schedules/upcoming/ - 다가오는 스케줄
- POST /api/v1/analytics/push/subscribe/ - Web Push 구독 등록
- GET  /api/v1/analytics/push/vapid-key/ - VAPID 공개키 조회
"""

import logging
from datetime import datetime
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
from django.utils import timezone

logger = logging.getLogger(__name__)


class ScheduleListView(APIView):
    """복습 스케줄 목록 조회 및 생성"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="복습 스케줄 목록 조회",
        description="""
사용자의 복습 스케줄 목록을 조회합니다.

### 필터링 옵션
- `status`: PENDING, NOTIFIED, COMPLETED, SKIPPED
- `include_past`: 과거 스케줄 포함 여부
        """,
        parameters=[
            OpenApiParameter(name='user_id', type=OpenApiTypes.UUID, required=True),
            OpenApiParameter(name='status', type=OpenApiTypes.STR, required=False),
            OpenApiParameter(name='include_past', type=OpenApiTypes.BOOL, required=False, default=False),
            OpenApiParameter(name='limit', type=OpenApiTypes.INT, required=False, default=50),
        ],
        responses={200: {'type': 'object'}}
    )
    def get(self, request):
        from services.scheduling.review_scheduler import ReviewScheduleService
        
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response(
                {"error": "user_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        status_filter = request.query_params.get('status')
        include_past = request.query_params.get('include_past', 'false').lower() == 'true'
        limit = int(request.query_params.get('limit', 50))
        
        try:
            service = ReviewScheduleService()
            schedules = service.get_user_schedules(
                user_id=user_id,
                status=status_filter,
                include_past=include_past,
                limit=limit
            )
            
            return Response({
                "user_id": user_id,
                "count": len(schedules),
                "schedules": schedules
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Schedule list error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @extend_schema(
        tags=['Analytics'],
        summary="수동 복습 스케줄 생성",
        description="""
복습 스케줄을 수동으로 생성합니다.

### 요청 예시
```json
{
    "user_id": "uuid",
    "scheduled_at": "2024-01-15T14:00:00+09:00",
    "domain": "cs",
    "note": "SQL 복습"
}
```
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string', 'format': 'uuid'},
                    'scheduled_at': {'type': 'string', 'format': 'date-time'},
                    'domain': {'type': 'string', 'enum': ['cs', 'dialect', 'all'], 'default': 'all'},
                    'node_id': {'type': 'string', 'format': 'uuid'},
                    'note': {'type': 'string'},
                },
                'required': ['user_id', 'scheduled_at']
            }
        },
        responses={201: {'type': 'object'}}
    )
    def post(self, request):
        from services.scheduling.review_scheduler import ReviewScheduleService
        from dateutil.parser import parse as parse_datetime
        
        user_id = request.data.get('user_id')
        scheduled_at_str = request.data.get('scheduled_at')
        domain = request.data.get('domain', 'all')
        node_id = request.data.get('node_id')
        note = request.data.get('note', '')
        
        if not user_id or not scheduled_at_str:
            return Response(
                {"error": "user_id and scheduled_at are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # 시간 파싱
            scheduled_at = parse_datetime(scheduled_at_str)
            if timezone.is_naive(scheduled_at):
                scheduled_at = timezone.make_aware(scheduled_at)
            
            service = ReviewScheduleService()
            result = service.create_manual_schedule(
                user_id=user_id,
                scheduled_at=scheduled_at,
                domain=domain,
                node_id=node_id,
                note=note
            )
            
            return Response(result, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.exception(f"Schedule create error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ScheduleAutoCreateView(APIView):
    """분석 결과 기반 자동 스케줄 생성"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="분석 결과 기반 자동 스케줄 생성",
        description="""
벤치마크 분석 결과(망각 계수)를 기반으로 최적의 복습 스케줄을 자동 생성합니다.

### 요청 예시
```json
{
    "user_id": "uuid",
    "k_cs": 0.05,
    "k_dialect": 0.12,
    "target_retention": 0.8
}
```

### 응답
- 도메인별 복습 예정 시각과 시간 라벨 반환
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string', 'format': 'uuid'},
                    'k_cs': {'type': 'number', 'description': 'CS 도메인 망각 계수'},
                    'k_dialect': {'type': 'number', 'description': '사투리 도메인 망각 계수'},
                    'target_retention': {'type': 'number', 'default': 0.8},
                },
                'required': ['user_id', 'k_cs', 'k_dialect']
            }
        },
        responses={201: {'type': 'object'}}
    )
    def post(self, request):
        from services.scheduling.review_scheduler import ReviewScheduleService
        
        user_id = request.data.get('user_id')
        k_cs = request.data.get('k_cs')
        k_dialect = request.data.get('k_dialect')
        target_retention = request.data.get('target_retention', 0.8)
        
        if not user_id or k_cs is None or k_dialect is None:
            return Response(
                {"error": "user_id, k_cs, and k_dialect are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            service = ReviewScheduleService()
            result = service.create_schedule_from_analysis(
                user_id=user_id,
                k_cs=float(k_cs),
                k_dialect=float(k_dialect),
                target_retention=float(target_retention)
            )
            
            return Response(result, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.exception(f"Auto schedule create error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ScheduleDetailView(APIView):
    """스케줄 상세 조회/수정/삭제"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="복습 스케줄 상세 조회",
        parameters=[
            OpenApiParameter(name='schedule_id', type=OpenApiTypes.UUID, location=OpenApiParameter.PATH)
        ],
        responses={200: {'type': 'object'}}
    )
    def get(self, request, schedule_id):
        from analytics.schedule_models import ReviewSchedule
        
        try:
            schedule = ReviewSchedule.objects.select_related('user', 'node').get(id=schedule_id)
            
            return Response({
                "id": str(schedule.id),
                "user_id": str(schedule.user_id),
                "username": schedule.user.username,
                "domain": schedule.domain,
                "node_id": str(schedule.node_id) if schedule.node_id else None,
                "node_title": schedule.node.title if schedule.node else None,
                "scheduled_at": schedule.scheduled_at.isoformat(),
                "notified_at": schedule.notified_at.isoformat() if schedule.notified_at else None,
                "completed_at": schedule.completed_at.isoformat() if schedule.completed_at else None,
                "status": schedule.status,
                "status_display": schedule.get_status_display(),
                "target_retention": schedule.target_retention,
                "forgetting_k": schedule.forgetting_k,
                "is_manual": schedule.is_manual,
                "is_due": schedule.is_due,
                "time_until_due_minutes": round(schedule.time_until_due, 1),
                "note": schedule.note,
            }, status=status.HTTP_200_OK)
            
        except ReviewSchedule.DoesNotExist:
            return Response(
                {"error": f"Schedule not found: {schedule_id}"},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @extend_schema(
        tags=['Analytics'],
        summary="복습 스케줄 수정",
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'scheduled_at': {'type': 'string', 'format': 'date-time'},
                    'status': {'type': 'string', 'enum': ['PENDING', 'NOTIFIED', 'COMPLETED', 'SKIPPED']},
                    'note': {'type': 'string'},
                }
            }
        },
        responses={200: {'type': 'object'}}
    )
    def patch(self, request, schedule_id):
        from services.scheduling.review_scheduler import ReviewScheduleService
        from dateutil.parser import parse as parse_datetime
        
        scheduled_at_str = request.data.get('scheduled_at')
        status_value = request.data.get('status')
        note = request.data.get('note')
        
        scheduled_at = None
        if scheduled_at_str:
            scheduled_at = parse_datetime(scheduled_at_str)
            if timezone.is_naive(scheduled_at):
                scheduled_at = timezone.make_aware(scheduled_at)
        
        try:
            service = ReviewScheduleService()
            result = service.update_schedule(
                schedule_id=schedule_id,
                scheduled_at=scheduled_at,
                status=status_value,
                note=note
            )
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Schedule update error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @extend_schema(
        tags=['Analytics'],
        summary="복습 스케줄 삭제",
        responses={204: None}
    )
    def delete(self, request, schedule_id):
        from services.scheduling.review_scheduler import ReviewScheduleService
        
        try:
            service = ReviewScheduleService()
            deleted = service.delete_schedule(schedule_id)
            
            if deleted:
                return Response(status=status.HTTP_204_NO_CONTENT)
            else:
                return Response(
                    {"error": "Schedule not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
                
        except Exception as e:
            logger.exception(f"Schedule delete error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ScheduleCompleteView(APIView):
    """복습 완료 표시 및 망각 곡선 갱신"""
    
    # recall_state → is_correct 변환 매핑
    RECALL_STATE_MAPPING = {
        "REMEMBERED": True,
        "FORGOT": False,
    }
    
    @extend_schema(
        tags=['Analytics'],
        summary="복습 완료 및 결과 제출",
        description="""
해당 스케줄의 복습을 완료 처리하고, 결과를 바탕으로 망각 곡선을 갱신합니다.

### 요청 예시
```json
{
    "recall_state": "REMEMBERED",
    "confidence_score": 0.8,
    "response_time_ms": 1500
}
```

### recall_state
- `REMEMBERED`: 기억함
- `FORGOT`: 잊어버림

### 응답
- 갱신된 `forgetting_k` 및 다음 복습 예정 시간 반환
        """,
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'recall_state': {'type': 'string', 'enum': ['REMEMBERED', 'FORGOT']},
                    'confidence_score': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'response_time_ms': {'type': 'integer'},
                },
                'required': ['recall_state', 'confidence_score']
            }
        },
        responses={200: {'type': 'object'}}
    )
    def post(self, request, schedule_id):
        from analytics.schedule_models import ReviewSchedule
        from analytics.models import User
        
        try:
            schedule = ReviewSchedule.objects.get(id=schedule_id)
            user = schedule.user
            domain = schedule.domain
            
            # 복습 결과 파싱
            recall_state = request.data.get('recall_state')
            confidence_score = request.data.get('confidence_score', 0.5)
            response_time_ms = request.data.get('response_time_ms', 0)
            
            # recall_state 유효성 검증
            if recall_state and recall_state not in self.RECALL_STATE_MAPPING:
                return Response(
                    {"error": f"Invalid recall_state: {recall_state}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            is_correct = self.RECALL_STATE_MAPPING.get(recall_state, True)
            
            # 스케줄 완료 처리
            schedule.mark_completed()
            
            # =========================================================
            # 망각 곡선 갱신 로직
            # =========================================================
            domain_stat = user.get_domain_stat(domain)
            old_k = domain_stat.forgetting_k
            
            # 착각 지수 계산: confidence - 실제 정답 여부
            actual_score = 1.0 if is_correct else 0.0
            illusion_delta = confidence_score - actual_score
            
            # 망각 상수 조정
            # - 맞춤: k를 소폭 감소 (망각이 느려짐)
            # - 틀림: k를 소폭 증가 (망각이 빨라짐)
            adjustment_rate = 0.05  # 조정 폭
            if is_correct:
                new_k = old_k * (1 - adjustment_rate)
            else:
                new_k = old_k * (1 + adjustment_rate * 2)  # 틀리면 더 큰 폭으로 증가
            
            # 범위 제한 (0.01 ~ 5.0)
            new_k = max(0.01, min(new_k, 5.0))
            
            # DB 저장
            domain_stat.forgetting_k = new_k
            domain_stat.save(update_fields=['forgetting_k', 'updated_at'])
            
            # 사용자 illusion_avg 업데이트 (이동 평균)
            old_illusion = user.illusion_avg
            new_illusion = old_illusion * 0.9 + illusion_delta * 0.1
            user.illusion_avg = new_illusion
            user.save(update_fields=['illusion_avg'])
            
            # =========================================================
            # 다음 복습 시간 계산
            # =========================================================
            from services.cognitive.benchmark import calculate_next_review_hours
            next_review_hours = calculate_next_review_hours(
                k=new_k,
                target_retention=0.8,
                alpha=user.alpha_user,
                illusion=new_illusion
            )
            
            return Response({
                "id": str(schedule.id),
                "status": schedule.status,
                "completed_at": schedule.completed_at.isoformat(),
                "domain": domain,
                "review_result": {
                    "is_correct": is_correct,
                    "confidence_score": confidence_score,
                    "illusion_delta": round(illusion_delta, 3)
                },
                "updated_stats": {
                    "old_k": round(old_k, 4),
                    "new_k": round(new_k, 4),
                    "illusion_avg": round(new_illusion, 3)
                },
                "next_review": {
                    "hours": round(next_review_hours, 2),
                    "label": f"{next_review_hours:.1f}시간 후"
                }
            }, status=status.HTTP_200_OK)
            
        except ReviewSchedule.DoesNotExist:
            return Response(
                {"error": f"Schedule not found: {schedule_id}"},
                status=status.HTTP_404_NOT_FOUND
            )


class ScheduleCheckNotifyView(APIView):
    """알림 확인 및 전송"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="알림 확인 및 전송 (WebPush / MacOS)",
        description="""
현재 시점에 복습 시간이 된 스케줄을 확인하고 알림을 전송합니다.
Web Push 구독이 있는 경우 브라우저로, 로컬 서버인 경우 맥북으로 알림을 보냅니다.
        """,
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'checked_at': {'type': 'string'},
                    'due_count': {'type': 'integer'},
                    'notified': {'type': 'array'},
                    'failed': {'type': 'array'},
                }
            }
        }
    )
    def post(self, request):
        from services.scheduling.review_scheduler import ReviewNotificationScheduler
        
        try:
            scheduler = ReviewNotificationScheduler()
            result = scheduler.check_and_notify()
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Check notify error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ScheduleUpcomingView(APIView):
    """다가오는 복습 스케줄 조회"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="다가오는 복습 스케줄 조회",
        description="지정된 시간 내에 예정된 복습 스케줄을 조회합니다.",
        parameters=[
            OpenApiParameter(name='user_id', type=OpenApiTypes.UUID, required=False),
            OpenApiParameter(name='hours_ahead', type=OpenApiTypes.INT, required=False, default=24),
        ],
        responses={200: {'type': 'object'}}
    )
    def get(self, request):
        from services.scheduling.review_scheduler import ReviewNotificationScheduler
        
        user_id = request.query_params.get('user_id')
        hours_ahead = int(request.query_params.get('hours_ahead', 24))
        
        try:
            scheduler = ReviewNotificationScheduler()
            schedules = scheduler.get_upcoming_schedules(
                user_id=user_id,
                hours_ahead=hours_ahead
            )
            
            return Response({
                "hours_ahead": hours_ahead,
                "count": len(schedules),
                "schedules": schedules
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.exception(f"Upcoming schedules error: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PushSubscriptionView(APIView):
    """Web Push 구독 등록"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="Web Push 구독 등록",
        description="브라우저의 Push Subscription 정보를 서버에 등록합니다.",
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string', 'format': 'uuid'},
                    'endpoint': {'type': 'string'},
                    'keys': {
                        'type': 'object',
                        'properties': {
                            'p256dh': {'type': 'string'},
                            'auth': {'type': 'string'}
                        }
                    }
                },
                'required': ['user_id', 'endpoint', 'keys']
            }
        },
        responses={201: {'description': '구독 성공'}}
    )
    def post(self, request):
        from analytics.models import User
        from analytics.schedule_models import PushSubscription
        
        user_id = request.data.get('user_id')
        endpoint = request.data.get('endpoint')
        keys = request.data.get('keys', {})
        
        if not user_id or not endpoint or 'p256dh' not in keys or 'auth' not in keys:
             return Response(
                 {"error": "Invalid subscription data. 'user_id', 'endpoint', 'keys.p256dh', 'keys.auth' are required."},
                 status=status.HTTP_400_BAD_REQUEST
             )

        try:
            user = User.objects.get(id=user_id)
            
            # 구독 정보 저장 또는 갱신
            PushSubscription.objects.update_or_create(
                user=user,
                endpoint=endpoint,
                defaults={
                    'p256dh': keys['p256dh'],
                    'auth': keys['auth'],
                    'user_agent': request.META.get('HTTP_USER_AGENT', '')[:255]
                }
            )
            
            return Response({"status": "subscribed"}, status=status.HTTP_201_CREATED)
            
        except User.DoesNotExist:
             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.exception(f"구독 처리 오류: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class VapidKeyView(APIView):
    """VAPID 공개키 조회"""
    
    @extend_schema(
        tags=['Analytics'],
        summary="VAPID 공개키 조회",
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'publicKey': {'type': 'string'}
                }
            }
        }
    )
    def get(self, request):
        from django.conf import settings
        return Response({
            "publicKey": settings.VAPID_PUBLIC_KEY
        }, status=status.HTTP_200_OK)
