"""
Scheduling Celery Tasks - 복습 알림 백그라운드 태스크

주기적으로 복습 시간을 확인하고 알림을 전송합니다.
"""

import logging
from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(name='check_review_notifications')
def check_review_notifications():
    """
    복습 알림 체크 태스크
    
    1분마다 실행되어 복습 시간이 된 스케줄을 확인하고
    맥OS 알림을 전송합니다.
    """
    from services.scheduling.review_scheduler import ReviewNotificationScheduler
    
    try:
        scheduler = ReviewNotificationScheduler()
        result = scheduler.check_and_notify()
        
        if result["due_count"] > 0:
            logger.info(
                f"[ReviewNotification] 체크 완료 - "
                f"대기: {result['due_count']}, "
                f"알림 전송: {len(result['notified'])}, "
                f"실패: {len(result['failed'])}"
            )
        
        return result
        
    except Exception as e:
        logger.exception(f"[ReviewNotification] 태스크 오류: {e}")
        return {"error": str(e)}


@shared_task(name='cleanup_old_schedules')
def cleanup_old_schedules(days_old: int = 30):
    """
    오래된 완료/스킵된 스케줄 정리
    
    Args:
        days_old: 며칠 지난 스케줄을 삭제할지
    """
    from analytics.schedule_models import ReviewSchedule, ScheduleStatus
    from datetime import timedelta
    
    try:
        cutoff = timezone.now() - timedelta(days=days_old)
        
        deleted_count, _ = ReviewSchedule.objects.filter(
            status__in=[ScheduleStatus.COMPLETED, ScheduleStatus.SKIPPED],
            scheduled_at__lt=cutoff
        ).delete()
        
        if deleted_count > 0:
            logger.info(f"[Cleanup] {deleted_count}개의 오래된 스케줄 삭제됨")
        
        return {"deleted_count": deleted_count}
        
    except Exception as e:
        logger.exception(f"[Cleanup] 태스크 오류: {e}")
        return {"error": str(e)}


@shared_task(name='send_test_notification')
def send_test_notification(message: str = "테스트 알림입니다!"):
    """
    테스트 알림 전송
    
    Args:
        message: 알림 메시지
    """
    from services.scheduling.review_scheduler import MacOSNotificationService
    
    try:
        service = MacOSNotificationService()
        success = service.send_notification(
            title="📚 Memorylary 테스트",
            message=message,
            subtitle="테스트 알림"
        )
        
        return {"success": success, "message": message}
        
    except Exception as e:
        logger.exception(f"[TestNotification] 오류: {e}")
        return {"success": False, "error": str(e)}
