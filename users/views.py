from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.contrib.auth.models import User as AuthUser
from drf_spectacular.utils import extend_schema

from users.serializers import UserCreationSerializer, LoginSerializer
from analytics.models import User as AnalyticsUser, UserDomainStat

import logging
logger = logging.getLogger(__name__)

class RegisterView(APIView):
    """
    POST /api/v1/auth/register/
    회원가입 API
    """
    @extend_schema(
        tags=['Auth'],
        summary="회원가입",
        request=UserCreationSerializer,
        responses={201: UserCreationSerializer}
    )
    def post(self, request):
        try:
            logger.info(f"[Register] Request Data: {request.data}")
            # Request Body(request.data)에서 데이터 파싱 및 검증
            serializer = UserCreationSerializer(data=request.data)
            if serializer.is_valid():
                # 1. Django Auth User 생성
                user = serializer.save()
                
                # 2. Analytics User 생성 (username 공유)
                if not AnalyticsUser.objects.filter(username=user.username).exists():
                    logger.info(f"[Register] Creating AnalyticsUser for {user.username}")
                    AnalyticsUser.objects.create(
                        username=user.username,
                        alpha_user=1.0,  # 기본값
                        base_forgetting_k=0.5  # 기본값
                    )
                
                logger.info(f"[Register] Success: {user.username}")
                return Response({
                    "message": "User created successfully",
                    "user": serializer.data
                }, status=status.HTTP_201_CREATED)
            
            logger.warning(f"[Register] Invalid Data: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.exception(f"[Register] Error: {e}")
            raise e

class LoginView(APIView):
    """
    POST /api/v1/auth/login/
    로그인 API
    """
    @extend_schema(
        tags=['Auth'],
        summary="로그인",
        request=LoginSerializer,
        responses={
            200: {
                'type': 'object',
                'properties': {
                    'token': {'type': 'string'},
                    'user_id': {'type': 'string'},
                    'username': {'type': 'string'},
                    'alpha': {'type': 'number'},
                    'forgetting_k': {'type': 'number'},
                    'domain_stats': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'domain': {'type': 'string'},
                                'alpha': {'type': 'number'},
                                'forgetting_k': {'type': 'number'}
                            }
                        }
                    }
                }
            }
        }
    )
    def post(self, request):
        try:
            logger.info(f"[Login] Request: username={request.data.get('username')}")
            # 1. Serializer를 통해 Body 데이터 검증
            serializer = LoginSerializer(data=request.data)
            if not serializer.is_valid():
                logger.warning(f"[Login] Invalid Data: {serializer.errors}")
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
                
            username = serializer.validated_data['username']
            password = serializer.validated_data['password']
            
            user = authenticate(username=username, password=password)
            
            if user:
                logger.info(f"[Login] Authenticated: {username}")
                # 2. 토큰 발급
                token, _ = Token.objects.get_or_create(user=user)
                
                # 3. Analytics User 정보 조회
                try:
                    analytics_user = AnalyticsUser.objects.get(username=username)
                except AnalyticsUser.DoesNotExist:
                    logger.warning(f"[Login] AnalyticsUser missing for {username}, creating default.")
                    # 없으면 생성 (방어 로직)
                    analytics_user = AnalyticsUser.objects.create(
                        username=username,
                        alpha_user=1.0,
                        base_forgetting_k=0.5
                    )
                
                # 4. 도메인별 망각계수 조회
                domain_stats = UserDomainStat.objects.filter(user=analytics_user)
                stats_data = [
                    {
                        "domain": stat.domain,
                        "alpha": stat.alpha,
                        "forgetting_k": stat.forgetting_k
                    }
                    for stat in domain_stats
                ]
                
                logger.info(f"[Login] Success: {username}, Token: {token.key[:10]}...")
                return Response({
                    "token": token.key,
                    "user_id": user.pk,  # Auth User ID
                    "uuid": str(analytics_user.id), # Analytics User UUID
                    "username": user.username,
                    "alpha": analytics_user.alpha_user,
                    "forgetting_k": analytics_user.base_forgetting_k,
                    "domain_stats": stats_data
                }, status=status.HTTP_200_OK)
            
            logger.warning(f"[Login] Authentication Failed: {username}")
            return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.exception(f"[Login] Error: {e}")
            raise e
