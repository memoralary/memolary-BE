"""
Analytics URL Configuration

벤치마크 API 엔드포인트:
- POST /api/benchmark/initialize - 벤치마크 초기화
- POST /api/benchmark/submit - 테스트 결과 제출
- POST /api/benchmark/analyze - 결과 분석
- GET /api/benchmark/status/<user_id> - 진행 상태 확인
"""

from django.urls import path
from analytics.views import (
    BenchmarkInitializeView,
    BenchmarkSubmitView,
    BenchmarkAnalyzeView,
    BenchmarkStatusView,
)

app_name = 'analytics'

urlpatterns = [
    # 벤치마크 API
    path('benchmark/initialize/', BenchmarkInitializeView.as_view(), name='benchmark-initialize'),
    path('benchmark/submit/', BenchmarkSubmitView.as_view(), name='benchmark-submit'),
    path('benchmark/analyze/', BenchmarkAnalyzeView.as_view(), name='benchmark-analyze'),
    path('benchmark/status/<uuid:user_id>/', BenchmarkStatusView.as_view(), name='benchmark-status'),
]
