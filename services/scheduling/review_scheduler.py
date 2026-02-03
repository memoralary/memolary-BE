"""
Review Scheduling Service - 복습 스케줄 관리 및 알림 서비스

기능:
- 테스트 결과 기반 복습 스케줄 자동 생성
- 수동 복습 스케줄 설정
- 맥OS 알림 전송
- 스케줄 모니터링
"""

import logging
import subprocess
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from django.utils import timezone
from django.db.models import Q

logger = logging.getLogger(__name__)


class ReviewScheduleService:
    """복습 스케줄 관리 서비스"""
    
    DEFAULT_TARGET_RETENTION = 0.8
    
    def __init__(self):
        from services.cognitive.benchmark import ReviewScheduleCalculator
        self.calculator = ReviewScheduleCalculator()
    
    def get_user_schedules(
        self,
        user_id: str,
        status: str = None,
        include_past: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        사용자의 복습 스케줄 조회
        
        Args:
            user_id: 사용자 ID
            status: 필터링할 상태 (PENDING, NOTIFIED, COMPLETED, SKIPPED)
            include_past: 과거 스케줄 포함 여부
            limit: 최대 조회 수
            
        Returns:
            스케줄 목록
        """
        from analytics.schedule_models import ReviewSchedule, ScheduleStatus
        
        queryset = ReviewSchedule.objects.filter(user_id=user_id)
        
        if status:
            queryset = queryset.filter(status=status)
        
        if not include_past:
            queryset = queryset.filter(
                Q(status=ScheduleStatus.PENDING) |
                Q(scheduled_at__gte=timezone.now() - timedelta(hours=1))
            )
        
        schedules = queryset.select_related('node').order_by('scheduled_at')[:limit]
        
        return [
            {
                "id": str(s.id),
                "domain": s.domain,
                "node_id": str(s.node_id) if s.node_id else None,
                "node_title": s.node.title if s.node else None,
                "scheduled_at": s.scheduled_at.isoformat(),
                "scheduled_at_local": s.scheduled_at.strftime("%Y-%m-%d %H:%M"),
                "status": s.status,
                "status_display": s.get_status_display(),
                "target_retention": s.target_retention,
                "forgetting_k": s.forgetting_k,
                "is_manual": s.is_manual,
                "is_due": s.is_due,
                "time_until_due_minutes": round(s.time_until_due, 1),
                "note": s.note,
            }
            for s in schedules
        ]
    
    def create_schedule_from_analysis(
        self,
        user_id: str,
        k_cs: float,
        k_dialect: float,
        target_retention: float = None,
        from_time: datetime = None
    ) -> Dict[str, Any]:
        """
        분석 결과 기반 복습 스케줄 자동 생성
        
        Args:
            user_id: 사용자 ID
            k_cs: CS 도메인 망각 계수
            k_dialect: 사투리 도메인 망각 계수
            target_retention: 목표 암기율
            from_time: 기준 시각 (기본: 현재)
            
        Returns:
            생성된 스케줄 정보
        """
        from analytics.models import User
        from analytics.schedule_models import ReviewSchedule
        
        target_retention = target_retention or self.DEFAULT_TARGET_RETENTION
        from_time = from_time or timezone.now()
        
        user = User.objects.get(id=user_id)
        
        # 복습 스케줄 계산
        schedule = self.calculator.calculate_review_schedule(
            k_cs=k_cs,
            k_dialect=k_dialect,
            target_retention=target_retention,
            from_time=from_time
        )
        
        created_schedules = []
        
        # CS 도메인 스케줄
        if schedule.cs_review_datetime:
            cs_schedule = ReviewSchedule.objects.create(
                user=user,
                domain='cs',
                scheduled_at=schedule.cs_review_datetime,
                target_retention=target_retention,
                forgetting_k=k_cs,
                is_manual=False,
                note=f"자동 생성 - 목표 암기율 {target_retention*100:.0f}%"
            )
            created_schedules.append({
                "id": str(cs_schedule.id),
                "domain": "cs",
                "scheduled_at": schedule.cs_review_datetime.isoformat(),
                "hours_from_now": schedule.cs_review_hours,
                "label": self.calculator.format_hours_to_human_readable(schedule.cs_review_hours)
            })
        
        # Dialect 도메인 스케줄
        if schedule.dialect_review_datetime:
            dialect_schedule = ReviewSchedule.objects.create(
                user=user,
                domain='dialect',
                scheduled_at=schedule.dialect_review_datetime,
                target_retention=target_retention,
                forgetting_k=k_dialect,
                is_manual=False,
                note=f"자동 생성 - 목표 암기율 {target_retention*100:.0f}%"
            )
            created_schedules.append({
                "id": str(dialect_schedule.id),
                "domain": "dialect",
                "scheduled_at": schedule.dialect_review_datetime.isoformat(),
                "hours_from_now": schedule.dialect_review_hours,
                "label": self.calculator.format_hours_to_human_readable(schedule.dialect_review_hours)
            })
        
        return {
            "user_id": str(user_id),
            "target_retention": target_retention,
            "created_count": len(created_schedules),
            "schedules": created_schedules
        }
    
    def create_manual_schedule(
        self,
        user_id: str,
        scheduled_at: datetime,
        domain: str = 'all',
        node_id: str = None,
        note: str = ''
    ) -> Dict[str, Any]:
        """
        수동 복습 스케줄 생성
        
        Args:
            user_id: 사용자 ID
            scheduled_at: 복습 예정 시각
            domain: 도메인 (cs, dialect, all)
            node_id: 특정 노드 ID (선택)
            note: 메모
            
        Returns:
            생성된 스케줄 정보
        """
        from analytics.models import User
        from analytics.schedule_models import ReviewSchedule
        from knowledge.models import KnowledgeNode
        
        user = User.objects.get(id=user_id)
        
        node = None
        if node_id:
            node = KnowledgeNode.objects.get(id=node_id)
        
        schedule = ReviewSchedule.objects.create(
            user=user,
            node=node,
            domain=domain,
            scheduled_at=scheduled_at,
            is_manual=True,
            note=note or "수동 설정"
        )
        
        return {
            "id": str(schedule.id),
            "user_id": str(user_id),
            "domain": domain,
            "node_id": str(node_id) if node_id else None,
            "scheduled_at": scheduled_at.isoformat(),
            "scheduled_at_local": scheduled_at.strftime("%Y-%m-%d %H:%M"),
            "is_manual": True,
            "note": schedule.note
        }
    
    def update_schedule(
        self,
        schedule_id: str,
        scheduled_at: datetime = None,
        status: str = None,
        note: str = None
    ) -> Dict[str, Any]:
        """스케줄 수정"""
        from analytics.schedule_models import ReviewSchedule
        
        schedule = ReviewSchedule.objects.get(id=schedule_id)
        
        if scheduled_at:
            schedule.scheduled_at = scheduled_at
        if status:
            schedule.status = status
        if note is not None:
            schedule.note = note
        
        schedule.save()
        
        return {
            "id": str(schedule.id),
            "scheduled_at": schedule.scheduled_at.isoformat(),
            "status": schedule.status,
            "note": schedule.note
        }
    
    def delete_schedule(self, schedule_id: str) -> bool:
        """스케줄 삭제"""
        from analytics.schedule_models import ReviewSchedule
        
        try:
            schedule = ReviewSchedule.objects.get(id=schedule_id)
            schedule.delete()
            return True
        except ReviewSchedule.DoesNotExist:
            return False
    
    def get_due_schedules(self) -> List:
        """현재 시점에 알림이 필요한 스케줄 조회"""
        from analytics.schedule_models import ReviewSchedule, ScheduleStatus
        
        return list(
            ReviewSchedule.objects.filter(
                status=ScheduleStatus.PENDING,
                scheduled_at__lte=timezone.now()
            ).select_related('user', 'node')
        )


class MacOSNotificationService:
    """맥OS 알림 서비스"""
    
    def send_notification(
        self,
        title: str,
        message: str,
        subtitle: str = "",
        sound: str = "default"
    ) -> bool:
        """
        맥OS 알림 전송
        
        Args:
            title: 알림 제목
            message: 알림 내용
            subtitle: 부제목
            sound: 알림 소리
            
        Returns:
            성공 여부
        """
        try:
            # AppleScript를 사용한 알림
            script = f'''
            display notification "{message}" with title "{title}" subtitle "{subtitle}" sound name "{sound}"
            '''
            
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"[MacOS Notification] 전송 성공: {title}")
                return True
            else:
                logger.error(f"[MacOS Notification] 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("[MacOS Notification] 타임아웃")
            return False
        except Exception as e:
            logger.exception(f"[MacOS Notification] 오류: {e}")
            return False
    
    def send_review_reminder(
        self,
        username: str,
        domain: str,
        node_title: str = None
    ) -> bool:
        """
        복습 알림 전송
        
        Args:
            username: 사용자 이름
            domain: 도메인 (cs, dialect)
            node_title: 노드 제목 (선택)
        """
        domain_display = {
            'cs': 'CS 지식',
            'dialect': '사투리',
            'all': '전체'
        }.get(domain, domain)
        
        title = "📚 복습 시간이에요!"
        
        if node_title:
            message = f"{username}님, '{node_title}' 복습할 시간입니다."
        else:
            message = f"{username}님, {domain_display} 복습할 시간입니다."
        
        subtitle = "Memorylary"
        
        return self.send_notification(title, message, subtitle)


class ReviewNotificationScheduler:
    """
    복습 알림 스케줄러
    
    백그라운드에서 실행되어 복습 시간이 된 스케줄에 알림을 전송합니다.
    """
    
    def __init__(self):
        self.schedule_service = ReviewScheduleService()
        self.notification_service = MacOSNotificationService()
    
    def check_and_notify(self) -> Dict[str, Any]:
        """
        알림이 필요한 스케줄 확인 및 알림 전송
        
        Returns:
            처리 결과
        """
        from analytics.schedule_models import NotificationLog
        
        due_schedules = self.schedule_service.get_due_schedules()
        
        results = {
            "checked_at": timezone.now().isoformat(),
            "due_count": len(due_schedules),
            "notified": [],
            "failed": []
        }
        
        for schedule in due_schedules:
            try:
                # 알림 전송
                success = self.notification_service.send_review_reminder(
                    username=schedule.user.username,
                    domain=schedule.domain,
                    node_title=schedule.node.title if schedule.node else None
                )
                
                # 로그 기록
                NotificationLog.objects.create(
                    schedule=schedule,
                    notification_type='macos',
                    success=success
                )
                
                if success:
                    # 스케줄 상태 업데이트
                    schedule.mark_notified()
                    results["notified"].append({
                        "schedule_id": str(schedule.id),
                        "user": schedule.user.username,
                        "domain": schedule.domain
                    })
                else:
                    results["failed"].append({
                        "schedule_id": str(schedule.id),
                        "error": "Notification send failed"
                    })
                    
            except Exception as e:
                logger.exception(f"알림 처리 오류: {e}")
                results["failed"].append({
                    "schedule_id": str(schedule.id),
                    "error": str(e)
                })
        
        return results
    
    def get_upcoming_schedules(
        self,
        user_id: str = None,
        hours_ahead: int = 24
    ) -> List[Dict[str, Any]]:
        """
        다가오는 복습 스케줄 조회
        
        Args:
            user_id: 특정 사용자 (선택)
            hours_ahead: 몇 시간 앞까지 조회
        """
        from analytics.schedule_models import ReviewSchedule, ScheduleStatus
        
        now = timezone.now()
        until = now + timedelta(hours=hours_ahead)
        
        queryset = ReviewSchedule.objects.filter(
            status=ScheduleStatus.PENDING,
            scheduled_at__gte=now,
            scheduled_at__lte=until
        )
        
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        
        schedules = queryset.select_related('user', 'node').order_by('scheduled_at')
        
        return [
            {
                "schedule_id": str(s.id),
                "user": s.user.username,
                "domain": s.domain,
                "node_title": s.node.title if s.node else None,
                "scheduled_at": s.scheduled_at.isoformat(),
                "minutes_until": round((s.scheduled_at - now).total_seconds() / 60)
            }
            for s in schedules
        ]
