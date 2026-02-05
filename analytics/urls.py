"""
Analytics URL Configuration

벤치마크 API 엔드포인트:
- POST /api/benchmark/initialize - 벤치마크 초기화
- POST /api/benchmark/submit - 테스트 결과 제출
- POST /api/benchmark/analyze - 결과 분석
- GET /api/benchmark/status/<user_id> - 진행 상태 확인

복습 스케줄 API 엔드포인트:
- GET/POST /api/v1/analytics/schedules/ - 스케줄 목록/생성
- POST /api/v1/analytics/schedules/auto/ - 자동 스케줄 생성
- GET/PATCH/DELETE /api/v1/analytics/schedules/<id>/ - 스케줄 상세
- POST /api/v1/analytics/schedules/<id>/complete/ - 복습 완료
- POST /api/v1/analytics/schedules/check-notify/ - 알림 확인/전송
- GET /api/v1/analytics/schedules/upcoming/ - 다가오는 스케줄

Web Push API 엔드포인트:
- POST /api/v1/analytics/push/subscribe/ - 구독 등록
- GET /api/v1/analytics/push/vapid-key/ - 공개키 조회
"""

from django.urls import path
from analytics.views import (
    BenchmarkInitializeView,
    BenchmarkSubmitView,
    BenchmarkAnalyzeView,
    BenchmarkStatusView,
    UserAnalysisResultView,
)
from analytics.schedule_views import (
    ScheduleListView,
    ScheduleAutoCreateView,
    ScheduleDetailView,
    ScheduleCompleteView,
    ScheduleCheckNotifyView,
    ScheduleUpcomingView,
    PushSubscriptionView,
    VapidKeyView,
)

app_name = 'analytics'

urlpatterns = [
    # 벤치마크 API
    path('benchmark/initialize/', BenchmarkInitializeView.as_view(), name='benchmark-initialize'),
    path('benchmark/submit/', BenchmarkSubmitView.as_view(), name='benchmark-submit'),
    path('benchmark/analyze/', BenchmarkAnalyzeView.as_view(), name='benchmark-analyze'),
    path('benchmark/status/<str:user_id>/', BenchmarkStatusView.as_view(), name='benchmark-status'),
    path('results/<str:user_id>/', UserAnalysisResultView.as_view(), name='user-analysis-result'),
    
    # 복습 스케줄 API
    path('schedules/', ScheduleListView.as_view(), name='schedule-list'),
    path('schedules/auto/', ScheduleAutoCreateView.as_view(), name='schedule-auto-create'),
    path('schedules/check-notify/', ScheduleCheckNotifyView.as_view(), name='schedule-check-notify'),
    path('schedules/upcoming/', ScheduleUpcomingView.as_view(), name='schedule-upcoming'),
    path('schedules/<str:schedule_id>/', ScheduleDetailView.as_view(), name='schedule-detail'),
    path('schedules/<str:schedule_id>/complete/', ScheduleCompleteView.as_view(), name='schedule-complete'),
    
    # Web Push API
    path('push/subscribe/', PushSubscriptionView.as_view(), name='push-subscribe'),
    path('push/vapid-key/', VapidKeyView.as_view(), name='push-vapid-key'),
]
