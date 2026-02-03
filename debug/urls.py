"""
Debug App URL Configuration

디버깅 API 엔드포인트:
- GET  /api/v1/debug/health/ - 시스템 상태 확인
- GET  /api/v1/debug/stats/ - DB 통계
- GET  /api/v1/debug/data/<model_name>/ - 모델 데이터 조회
- POST /api/v1/debug/seed/ - 테스트 데이터 삽입
- POST /api/v1/debug/clear/ - DB 데이터 삭제
- POST /api/v1/debug/reset/ - 전체 초기화 (clear + seed)
- POST /api/v1/debug/benchmark/quick/ - 벤치마크 빠른 테스트
"""

from django.urls import path

from debug.views import (
    HealthCheckView,
    StatsView,
    DataView,
    SeedDataView,
    ClearDataView,
    ResetView,
    QuickBenchmarkView,
)

urlpatterns = [
    # 시스템 상태
    path('health/', HealthCheckView.as_view(), name='debug-health'),
    path('stats/', StatsView.as_view(), name='debug-stats'),
    
    # 데이터 조회
    path('data/<str:model_name>/', DataView.as_view(), name='debug-data'),
    
    # 데이터 관리
    path('seed/', SeedDataView.as_view(), name='debug-seed'),
    path('clear/', ClearDataView.as_view(), name='debug-clear'),
    path('reset/', ResetView.as_view(), name='debug-reset'),
    
    # 테스트 유틸리티
    path('benchmark/quick/', QuickBenchmarkView.as_view(), name='debug-benchmark-quick'),
]
