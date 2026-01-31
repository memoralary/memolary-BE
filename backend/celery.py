"""
Celery 앱 설정

사용법:
    # 워커 실행
    celery -A backend worker -l info
    
    # 비트 스케줄러 (선택)
    celery -A backend beat -l info
"""

import os
from celery import Celery

# Django 설정 모듈 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# Celery 앱 생성
app = Celery('backend')

# Django 설정에서 Celery 설정 로드
app.config_from_object('django.conf:settings', namespace='CELERY')

# 모든 등록된 앱에서 태스크 자동 검색
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
