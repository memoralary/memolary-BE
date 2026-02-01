"""
macOS 복습 알림 서비스

사용법:
    python manage.py review_notifier --user_id <uuid>

기능:
    - 사용자의 복습 시간을 모니터링
    - 복습 시간이 되면 macOS Notification Center에 알림 표시
    - 도메인별(CS/사투리) 개별 알림 지원
"""

import os
import subprocess
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from analytics.models import User, TestSession, TestResult
from services.cognitive.benchmark import (
    ForgettingCurveAnalyzer,
    ReviewScheduleCalculator,
    calculate_next_review_hours,
)

logger = logging.getLogger(__name__)


def send_macos_notification(
    title: str,
    message: str,
    subtitle: str = "",
    sound: str = "default"
) -> bool:
    """
    macOS Notification Center에 알림 전송
    
    Args:
        title: 알림 제목
        message: 알림 본문
        subtitle: 알림 부제목 (선택)
        sound: 알림 사운드 ("default", "Basso", "Blow" 등)
        
    Returns:
        성공 여부
    """
    # AppleScript를 사용하여 알림 전송
    script_parts = [
        f'display notification "{message}"',
        f'with title "{title}"',
    ]
    
    if subtitle:
        script_parts.append(f'subtitle "{subtitle}"')
    
    if sound:
        script_parts.append(f'sound name "{sound}"')
    
    script = " ".join(script_parts)
    
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True
        )
        logger.info(f"[NOTIFY] 알림 전송: {title} - {message}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[NOTIFY] 알림 전송 실패: {e}")
        return False


def send_macos_popup(
    title: str,
    message: str,
    buttons: list = None,
    open_url: str = None
) -> Optional[str]:
    """
    macOS 팝업 다이얼로그 표시
    
    Args:
        title: 다이얼로그 제목
        message: 다이얼로그 본문
        buttons: 버튼 목록 (기본: ["나중에", "복습 시작"])
        open_url: "복습 시작" 클릭 시 열 URL (기본: None)
        
    Returns:
        클릭된 버튼 이름 또는 None
    """
    if buttons is None:
        buttons = ["나중에", "복습 시작"]
    
    buttons_str = ", ".join([f'"{b}"' for b in buttons])
    
    # 기본 팝업 스크립트
    script = f'display dialog "{message}" with title "{title}" buttons {{{buttons_str}}} default button "{buttons[-1]}"'
    
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True
        )
        # 결과 파싱: "button returned:복습 시작"
        output = result.stdout.strip()
        clicked_button = None
        if "button returned:" in output:
            clicked_button = output.split("button returned:")[1]
        
        # "복습 시작" 클릭 시 브라우저 열기
        if clicked_button == "복습 시작" and open_url:
            subprocess.run(
                ["open", "-a", "Google Chrome", open_url],
                check=True
            )
            logger.info(f"[POPUP] Chrome 열기: {open_url}")
        
        return clicked_button
        
    except subprocess.CalledProcessError as e:
        # 사용자가 취소한 경우 (ESC 또는 X 버튼)
        if e.returncode == 1:
            logger.info("[POPUP] 사용자가 팝업을 취소함")
            return None
        logger.error(f"[POPUP] 팝업 표시 실패: {e}")
        return None


class Command(BaseCommand):
    help = "복습 시간 알림 서비스 (macOS Notification Center 연동)"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--user_id',
            type=str,
            required=True,
            help='모니터링할 사용자 ID (UUID)'
        )
        parser.add_argument(
            '--target_retention',
            type=float,
            default=0.8,
            help='목표 암기율 (기본: 0.8)'
        )
        parser.add_argument(
            '--check_interval',
            type=int,
            default=60,
            help='확인 주기 (초, 기본: 60)'
        )
        parser.add_argument(
            '--popup',
            action='store_true',
            help='알림 대신 팝업 다이얼로그 사용'
        )
        parser.add_argument(
            '--once',
            action='store_true',
            help='한 번만 확인하고 종료 (테스트용)'
        )
    
    def handle(self, *args, **options):
        user_id = options['user_id']
        target_retention = options['target_retention']
        check_interval = options['check_interval']
        use_popup = options['popup']
        run_once = options['once']
        
        # 사용자 확인
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise CommandError(f"사용자를 찾을 수 없습니다: {user_id}")
        
        self.stdout.write(
            self.style.SUCCESS(f"🔔 복습 알림 서비스 시작: {user.username}")
        )
        self.stdout.write(f"   목표 암기율: {target_retention}")
        self.stdout.write(f"   확인 주기: {check_interval}초")
        self.stdout.write(f"   알림 방식: {'팝업' if use_popup else '알림센터'}")
        self.stdout.write("")
        
        # 알림 추적 (중복 방지)
        notified_domains = set()
        
        # 분석기 및 스케줄러
        analyzer = ForgettingCurveAnalyzer()
        scheduler = ReviewScheduleCalculator()
        
        try:
            while True:
                now = timezone.now()
                self.stdout.write(f"[{now.strftime('%H:%M:%S')}] 복습 시간 확인 중...")
                
                # 사용자의 최근 분석 결과 조회
                review_info = self._check_review_time(
                    user, target_retention, notified_domains
                )
                
                if review_info:
                    domain = review_info['domain']
                    hours_left = review_info['hours_left']
                    
                    if hours_left <= 0:
                        # 복습 시간 도래
                        title = "📚 복습 시간입니다!"
                        message = f"{domain} 도메인 복습이 필요합니다."
                        subtitle = f"목표 암기율 {int(target_retention*100)}% 유지를 위해"
                        
                        if use_popup:
                            # 복습 시작 버튼 클릭 시 Chrome 열기
                            review_url = f"http://localhost:3000/review?domain={domain}"
                            clicked = send_macos_popup(
                                title, 
                                f"{message}\n\n{subtitle}",
                                open_url=review_url
                            )
                            if clicked == "복습 시작":
                                self.stdout.write(
                                    self.style.SUCCESS(f"   🌐 Chrome 열림: {review_url}")
                                )
                        else:
                            send_macos_notification(title, message, subtitle)
                        
                        notified_domains.add(domain)
                        self.stdout.write(
                            self.style.WARNING(f"   ⏰ {domain} 복습 알림 발송!")
                        )
                    else:
                        hours = int(hours_left)
                        minutes = int((hours_left - hours) * 60)
                        self.stdout.write(
                            f"   {domain}: {hours}시간 {minutes}분 후 복습"
                        )
                
                if run_once:
                    break
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.stdout.write(self.style.SUCCESS("\n🛑 알림 서비스 종료"))
    
    def _check_review_time(self, user, target_retention, notified_domains):
        """
        복습 시간 확인
        
        Returns:
            {'domain': str, 'hours_left': float} 또는 None
        """
        # 사용자의 벤치마크 결과에서 k 값 조회
        # (실제 구현에서는 User 모델에 저장된 k 값을 사용)
        k_cs = getattr(user, 'k_cs', None) or 0.01
        k_dialect = getattr(user, 'k_dialect', None) or 0.15
        
        # 마지막 세션 시간 조회
        last_session = TestSession.objects.filter(
            user=user,
            performed_at__isnull=False
        ).order_by('-performed_at').first()
        
        if not last_session:
            return None
        
        # 마지막 세션으로부터 경과 시간
        now = timezone.now()
        elapsed_hours = (now - last_session.performed_at).total_seconds() / 3600
        
        # 각 도메인별 복습 필요 시간 계산
        cs_review_hours = calculate_next_review_hours(k_cs, target_retention)
        dialect_review_hours = calculate_next_review_hours(k_dialect, target_retention)
        
        # 더 급한 도메인 우선
        results = []
        
        if 'CS' not in notified_domains:
            cs_left = cs_review_hours - elapsed_hours
            results.append({'domain': 'CS', 'hours_left': cs_left})
        
        if '사투리' not in notified_domains:
            dialect_left = dialect_review_hours - elapsed_hours
            results.append({'domain': '사투리', 'hours_left': dialect_left})
        
        if not results:
            return None
        
        # 가장 급한 도메인 반환
        return min(results, key=lambda x: x['hours_left'])
