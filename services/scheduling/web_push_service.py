"""
Web Push Service - 웹 브라우저 푸시 알림 서비스

pywebpush 라이브러리를 사용하여 VAPID 방식의 푸시 알림을 전송합니다.
"""

import json
import logging
from django.conf import settings
from pywebpush import webpush, WebPushException
from analytics.schedule_models import PushSubscription

logger = logging.getLogger(__name__)


class WebPushService:
    """웹 푸시 알림 서비스"""
    
    def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        url: str = "/",
        tag: str = None
    ) -> int:
        """
        사용자의 모든 구독 기기로 푸시 알림 전송
        
        Args:
            user_id: 사용자 ID
            title: 알림 제목
            message: 알림 본문
            url: 클릭 시 이동할 URL
            tag: 알림 그룹화 태그
            
        Returns:
            전송 성공한 기기 수
        """
        if not settings.VAPID_PRIVATE_KEY:
            logger.warning("[WebPush] VAPID 키가 설정되지 않아 알림을 보낼 수 없습니다.")
            return 0
            
        subscriptions = PushSubscription.objects.filter(user_id=user_id)
        
        if not subscriptions.exists():
            # logger.debug(f"[WebPush] 구독 정보 없음: {user_id}")
            return 0
            
        # 페이로드 (Service Worker에서 처리할 데이터)
        payload = {
            "title": title,
            "body": message,
            "icon": "/logo192.png",  # 프론트엔드 에셋 경로
            "badge": "/badge.png",
            "data": {
                "url": url
            }
        }
        
        if tag:
            payload["tag"] = tag
            
        json_payload = json.dumps(payload)
        
        success_count = 0
        vapid_claims = {"sub": settings.VAPID_ADMIN_EMAIL}
        
        for sub in subscriptions:
            try:
                subscription_info = {
                    "endpoint": sub.endpoint,
                    "keys": {
                        "p256dh": sub.p256dh,
                        "auth": sub.auth
                    }
                }
                
                webpush(
                    subscription_info=subscription_info,
                    data=json_payload,
                    vapid_private_key=settings.VAPID_PRIVATE_KEY,
                    vapid_claims=vapid_claims
                )
                success_count += 1
                
                # 마지막 사용 시간 업데이트 (save 호출 시 auto_now 필드 갱신)
                sub.save()
                
            except WebPushException as ex:
                if ex.response and ex.response.status_code == 410:
                    logger.info(f"[WebPush] 만료된 구독 삭제: {sub.endpoint[:20]}")
                    sub.delete()
                else:
                    logger.error(f"[WebPush] 전송 실패: {ex}")
            except Exception as e:
                logger.exception(f"[WebPush] 처리 중 오류: {e}")
                
        return success_count
