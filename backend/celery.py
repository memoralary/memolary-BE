"""
Celery 앱 설정

사용법:
    # 워커 실행
    celery -A backend worker -l info
    
    # 비트 스케줄러 (주기적 알림 체크)
    celery -A backend beat -l info
    
    # 워커 + 비트 동시 실행 (개발용)
    celery -A backend worker -B -l info
"""

import os
from celery import Celery
from celery.schedules import crontab

# Django 설정 모듈 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# Celery 앱 생성
app = Celery('backend')

# Django 설정에서 Celery 설정 로드
app.config_from_object('django.conf:settings', namespace='CELERY')

# 모든 등록된 앱에서 태스크 자동 검색
app.autodiscover_tasks()


# =============================================================================
# 주기적 태스크 스케줄 (Celery Beat)
# =============================================================================
app.conf.beat_schedule = {
    # 1분마다 복습 알림 체크
    'check-review-notifications-every-minute': {
        'task': 'check_review_notifications',
        'schedule': 60.0,  # 60초마다
    },
    # 매일 새벽 3시에 오래된 스케줄 정리
    'cleanup-old-schedules-daily': {
        'task': 'cleanup_old_schedules',
        'schedule': crontab(hour=3, minute=0),
        'args': (30,),  # 30일 이상 된 스케줄 삭제
    },
}

app.conf.timezone = 'Asia/Seoul'


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
