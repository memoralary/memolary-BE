"""
Review Scheduling Service - 복습 스케줄 관리 및 알림 서비스

기능:
- 테스트 결과 기반 복습 스케줄 자동 생성
- 수동 복습 스케줄 설정
- 맥OS 알림 전송
- Web Push 알림 전송
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
        """사용자의 복습 스케줄 조회"""
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
        from_time: datetime = None,
        cs_node_ids: List[str] = None,
        dialect_node_ids: List[str] = None
    ) -> Dict[str, Any]:
        """분석 결과 기반 복습 스케줄 자동 생성 (노드별, 중복 방지)"""
        from analytics.models import User
        from analytics.schedule_models import ReviewSchedule, ScheduleStatus
        from knowledge.models import KnowledgeNode
        
        target_retention = target_retention or self.DEFAULT_TARGET_RETENTION
        from_time = from_time or timezone.now()
        cs_node_ids = cs_node_ids or []
        dialect_node_ids = dialect_node_ids or []
        
        user = User.objects.get(id=user_id)
        
        # 사용자의 학습 지능 및 메타인지 착각 가져오기
        alpha = user.alpha_user
        illusion = user.illusion_avg
        
        # 복습 스케줄 계산 (alpha, illusion 반영)
        schedule = self.calculator.calculate_review_schedule(
            k_cs=k_cs,
            k_dialect=k_dialect,
            target_retention=target_retention,
            from_time=from_time,
            alpha=alpha,
            illusion=illusion
        )
        
        created_schedules = []
        skipped_count = 0
        
        # =========================================================
        # CS 도메인 노드별 스케줄 생성
        # =========================================================
        for node_id in cs_node_ids:
            # 중복 체크: 해당 노드에 이미 PENDING 스케줄이 있으면 스킵
            existing = ReviewSchedule.objects.filter(
                user=user,
                node_id=node_id,
                status=ScheduleStatus.PENDING
            ).exists()
            
            if existing:
                skipped_count += 1
                continue
            
            try:
                node = KnowledgeNode.objects.get(id=node_id)
            except KnowledgeNode.DoesNotExist:
                continue
            
            cs_schedule = ReviewSchedule.objects.create(
                user=user,
                node=node,
                domain='cs',
                scheduled_at=schedule.cs_review_datetime,
                target_retention=target_retention,
                forgetting_k=k_cs,
                is_manual=False,
                note=f"자동 생성 - {node.title}"
            )
            created_schedules.append({
                "id": str(cs_schedule.id),
                "domain": "cs",
                "node_id": str(node.id),
                "node_title": node.title,
                "scheduled_at": schedule.cs_review_datetime.isoformat(),
                "hours_from_now": schedule.cs_review_hours,
                "label": self.calculator.format_hours_to_human_readable(schedule.cs_review_hours)
            })
        
        # =========================================================
        # Dialect 도메인 노드별 스케줄 생성
        # =========================================================
        for node_id in dialect_node_ids:
            existing = ReviewSchedule.objects.filter(
                user=user,
                node_id=node_id,
                status=ScheduleStatus.PENDING
            ).exists()
            
            if existing:
                skipped_count += 1
                continue
            
            try:
                node = KnowledgeNode.objects.get(id=node_id)
            except KnowledgeNode.DoesNotExist:
                continue
            
            dialect_schedule = ReviewSchedule.objects.create(
                user=user,
                node=node,
                domain='dialect',
                scheduled_at=schedule.dialect_review_datetime,
                target_retention=target_retention,
                forgetting_k=k_dialect,
                is_manual=False,
                note=f"자동 생성 - {node.title}"
            )
            created_schedules.append({
                "id": str(dialect_schedule.id),
                "domain": "dialect",
                "node_id": str(node.id),
                "node_title": node.title,
                "scheduled_at": schedule.dialect_review_datetime.isoformat(),
                "hours_from_now": schedule.dialect_review_hours,
                "label": self.calculator.format_hours_to_human_readable(schedule.dialect_review_hours)
            })
        
        # =========================================================
        # 노드 ID가 없는 경우: 도메인 레벨 스케줄 (기존 동작 유지)
        # =========================================================
        if not cs_node_ids and not dialect_node_ids:
            # 기존 PENDING 스케줄 체크
            cs_exists = ReviewSchedule.objects.filter(
                user=user, domain='cs', status=ScheduleStatus.PENDING, node__isnull=True
            ).exists()
            dialect_exists = ReviewSchedule.objects.filter(
                user=user, domain='dialect', status=ScheduleStatus.PENDING, node__isnull=True
            ).exists()
            
            if not cs_exists and schedule.cs_review_datetime:
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
            
            if not dialect_exists and schedule.dialect_review_datetime:
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
            "skipped_count": skipped_count,
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
        """수동 복습 스케줄 생성"""
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
        """맥OS 알림 전송"""
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


class ReviewNotificationScheduler:
    """
    복습 알림 스케줄러
    
    백그라운드에서 실행되어 복습 시간이 된 스케줄에 알림을 전송합니다.
    """
    
    def __init__(self):
        from services.scheduling.web_push_service import WebPushService
        self.schedule_service = ReviewScheduleService()
        self.macos_service = MacOSNotificationService()
        self.web_push_service = WebPushService()
    
    def check_and_notify(self) -> Dict[str, Any]:
        """
        알림이 필요한 스케줄 확인 및 알림 전송 (MacOS Native + Web Push)
        
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
                # 알림 메시지 구성
                username = schedule.user.username
                domain_display = {'cs': 'CS 지식', 'dialect': '사투리', 'all': '전체'}.get(schedule.domain, schedule.domain)
                
                title = "📚 복습 시간이에요!"
                message = f"{username}님, {domain_display} 복습할 시간입니다."
                if schedule.node:
                    message = f"{username}님, '{schedule.node.title}' 복습할 시간입니다."
                
                # 1. MacOS 알림 시도 (로컬 서버용 - Linux 서버에서는 동작 안함)
                # 에러 로그가 너무 많이 남지 않도록 try-catch 내부에서 처리
                macos_success = False
                try:
                    macos_success = self.macos_service.send_notification(title, message, "Memorylary")
                except Exception:
                    pass
                
                # 2. Web Push 알림 시도
                push_url = f"/review?schedule_id={schedule.id}"
                push_count = self.web_push_service.send_notification(
                    user_id=schedule.user_id,
                    title=title,
                    message=message,
                    url=push_url,
                    tag=f"review-{schedule.id}"
                )
                
                # 둘 중 하나라도 성공하면 성공 처리
                success = macos_success or (push_count > 0)
                
                noti_types = []
                if macos_success: noti_types.append('macos')
                if push_count > 0: noti_types.append('web_push')
                
                type_str = ','.join(noti_types) if noti_types else 'none'
                
                # 로그 기록
                NotificationLog.objects.create(
                    schedule=schedule,
                    notification_type=type_str,
                    success=success
                )
                
                if success:
                    # 스케줄 상태 업데이트
                    schedule.mark_notified()
                    results["notified"].append({
                        "schedule_id": str(schedule.id),
                        "user": username,
                        "methods": noti_types
                    })
                else:
                    results["failed"].append({
                        "schedule_id": str(schedule.id),
                        "error": "All notification methods failed"
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
