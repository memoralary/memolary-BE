"""
Review Schedule Models - 복습 스케줄 관리

사용자의 복습 일정을 저장하고 알림을 관리합니다.
"""

import uuid
from django.db import models
from django.utils import timezone


class ScheduleStatus(models.TextChoices):
    """스케줄 상태"""
    PENDING = 'PENDING', '대기중'
    NOTIFIED = 'NOTIFIED', '알림 전송됨'
    COMPLETED = 'COMPLETED', '완료'
    SKIPPED = 'SKIPPED', '건너뜀'


class ReviewSchedule(models.Model):
    """
    복습 스케줄 모델
    
    사용자의 개인화된 복습 일정을 관리합니다.
    망각곡선 기반으로 자동 생성되거나 수동으로 설정할 수 있습니다.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='스케줄 ID'
    )
    user = models.ForeignKey(
        'analytics.User',
        on_delete=models.CASCADE,
        related_name='review_schedules',
        verbose_name='사용자'
    )
    node = models.ForeignKey(
        'knowledge.KnowledgeNode',
        on_delete=models.CASCADE,
        related_name='review_schedules',
        verbose_name='복습 대상 노드',
        null=True,
        blank=True,
        help_text='특정 노드에 대한 복습 (null이면 전체 도메인)'
    )
    domain = models.CharField(
        max_length=20,
        default='all',
        verbose_name='도메인',
        help_text='cs, dialect, 또는 all'
    )
    
    # 시간 정보
    scheduled_at = models.DateTimeField(
        verbose_name='예정 시각',
        help_text='복습 예정 시각'
    )
    notified_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name='알림 시각',
        help_text='알림이 전송된 시각'
    )
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name='완료 시각'
    )
    
    # 스케줄 메타데이터
    status = models.CharField(
        max_length=20,
        choices=ScheduleStatus.choices,
        default=ScheduleStatus.PENDING,
        verbose_name='상태'
    )
    target_retention = models.FloatField(
        default=0.8,
        verbose_name='목표 암기율',
        help_text='0.0 ~ 1.0'
    )
    forgetting_k = models.FloatField(
        null=True,
        blank=True,
        verbose_name='망각 계수',
        help_text='이 스케줄 계산에 사용된 k값'
    )
    is_manual = models.BooleanField(
        default=False,
        verbose_name='수동 설정 여부'
    )
    note = models.TextField(
        blank=True,
        default='',
        verbose_name='메모'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = '복습 스케줄'
        verbose_name_plural = '복습 스케줄 목록'
        ordering = ['scheduled_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['scheduled_at']),
            models.Index(fields=['status', 'scheduled_at']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.domain} @ {self.scheduled_at}"
    
    @property
    def is_due(self) -> bool:
        """복습 시간이 되었는지 확인"""
        return self.status == ScheduleStatus.PENDING and self.scheduled_at <= timezone.now()
    
    @property
    def time_until_due(self) -> float:
        """복습까지 남은 시간 (분)"""
        if self.scheduled_at <= timezone.now():
            return 0
        delta = self.scheduled_at - timezone.now()
        return delta.total_seconds() / 60
    
    def mark_notified(self):
        """알림 전송 완료 표시"""
        self.status = ScheduleStatus.NOTIFIED
        self.notified_at = timezone.now()
        self.save(update_fields=['status', 'notified_at', 'updated_at'])
    
    def mark_completed(self):
        """복습 완료 표시"""
        self.status = ScheduleStatus.COMPLETED
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'completed_at', 'updated_at'])
    
    def mark_skipped(self):
        """복습 건너뛰기"""
        self.status = ScheduleStatus.SKIPPED
        self.save(update_fields=['status', 'updated_at'])


class NotificationLog(models.Model):
    """
    알림 로그 모델
    
    전송된 알림 내역을 기록합니다.
    """
    id = models.BigAutoField(primary_key=True)
    schedule = models.ForeignKey(
        ReviewSchedule,
        on_delete=models.CASCADE,
        related_name='notifications',
        verbose_name='관련 스케줄'
    )
    notification_type = models.CharField(
        max_length=50,
        default='macos',
        verbose_name='알림 유형',
        help_text='macos, email, push 등'
    )
    sent_at = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True, default='')
    
    class Meta:
        verbose_name = '알림 로그'
        verbose_name_plural = '알림 로그 목록'
        ordering = ['-sent_at']
    
    def __str__(self):
        return f"Notification for {self.schedule} @ {self.sent_at}"
