from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.contrib.auth.models import User as AuthUser
from drf_spectacular.utils import extend_schema

from users.serializers import UserCreationSerializer, LoginSerializer
from analytics.models import User as AnalyticsUser, UserDomainStat

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
        # Request Body(request.data)에서 데이터 파싱 및 검증
        serializer = UserCreationSerializer(data=request.data)
        if serializer.is_valid():
            # 1. Django Auth User 생성
            user = serializer.save()
            
            # 2. Analytics User 생성 (username 공유)
            if not AnalyticsUser.objects.filter(username=user.username).exists():
                AnalyticsUser.objects.create(
                    username=user.username,
                    alpha_user=1.0,  # 기본값
                    base_forgetting_k=0.5  # 기본값
                )
            
            return Response({
                "message": "User created successfully",
                "user": serializer.data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

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
        # 1. Serializer를 통해 Body 데이터 검증
        serializer = LoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        username = serializer.validated_data['username']
        password = serializer.validated_data['password']
        
        user = authenticate(username=username, password=password)
        
        if user:
            # 2. 토큰 발급
            token, _ = Token.objects.get_or_create(user=user)
            
            # 3. Analytics User 정보 조회
            try:
                analytics_user = AnalyticsUser.objects.get(username=username)
            except AnalyticsUser.DoesNotExist:
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
            
            return Response({
                "token": token.key,
                "user_id": user.pk,  # Auth User ID
                "uuid": str(analytics_user.id), # Analytics User UUID
                "username": user.username,
                "alpha": analytics_user.alpha_user,
                "forgetting_k": analytics_user.base_forgetting_k,
                "domain_stats": stats_data
            }, status=status.HTTP_200_OK)
        
        return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
