"""
Analytics Models - 인지 공학 기반 학습 분석

DBML 설계 기반 Django 모델:
- User: 사용자 전역 파라미터 (학습 지능 계수, 망각 상수, 메타인지 성향)
- TestSession: 실험/학습 세션 (시간 축 기반)
- TestResult: 인지 측정 결과 (핵심 관측 데이터)
- SpeechAnalysis: 설명형 심층 신호 분석 (A2 전용)
"""

import uuid
from django.db import models
from django.core.exceptions import ValidationError
from django.db.models import Avg

# 복습 스케줄 모델 (별도 파일에서 정의)
from analytics.schedule_models import ReviewSchedule, NotificationLog, ScheduleStatus


# =============================================================================
# Enums (TextChoices)
# =============================================================================

class TimePoint(models.TextChoices):
    """테스트 시점 - 에빙하우스 망각 곡선 기반"""
    T0 = 'T0', '직후 (0분)'
    T1 = 'T1', '20분 후'
    T2 = 'T2', '4시간 후'
    T3 = 'T3', '24시간 후'


class TestType(models.TextChoices):
    """테스트 유형 - 인지 부하 수준별 분류"""
    A1_BOTTOM_UP = 'A1_BOTTOM_UP', 'A1: 상향식 (개념→정의)'
    A2_TOP_DOWN = 'A2_TOP_DOWN', 'A2: 하향식 (정의→설명)'
    B_RECALL = 'B_RECALL', 'B: 자유 회상'


# =============================================================================
# [1] 사용자 모델 (Global Learning Intelligence)
# =============================================================================

class User(models.Model):
    """
    사용자 전역 파라미터 모델
    
    인지 공학 기반 학습 지능 지표:
    - alpha_user: 개인별 학습 지능 계수 (RT + 정답률 기반)
    - base_forgetting_k: 기초 망각 상수 k (초기 벤치마크 테스트)
    - illusion_avg: 평균 메타인지 착각 성향 (과신/신중 성향)
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='사용자 ID',
        help_text='사용자 고유 식별자 (UUID)'
    )
    username = models.CharField(
        max_length=150,
        unique=True,
        verbose_name='사용자명',
        help_text='로그인에 사용되는 고유 사용자명'
    )
    
    # 인지 공학 파라미터
    alpha_user = models.FloatField(
        default=1.0,
        verbose_name='학습 지능 계수',
        help_text='개인별 학습 지능 계수 α (RT + 정답률 기반, 기본값 1.0)'
    )
    base_forgetting_k = models.FloatField(
        default=0.5,
        verbose_name='기초 망각 상수',
        help_text='기초 망각 상수 k (초기 벤치마크 테스트 기반, 기본값 0.5)'
    )
    illusion_avg = models.FloatField(
        default=0.0,
        verbose_name='평균 메타인지 착각',
        help_text='평균 메타인지 착각 성향 (양수: 과신, 음수: 신중, 기본값 0.0)'
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='생성 시각',
        help_text='사용자 계정 생성 시각'
    )

    class Meta:
        verbose_name = '사용자'
        verbose_name_plural = '사용자 목록'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['username']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.username} (α={self.alpha_user:.2f})"
    
    def update_illusion_avg(self) -> float:
        """
        사용자의 모든 테스트 결과를 분석하여 illusion_avg 업데이트
        
        illusion_score = C - A (확신도 - 정답 여부)
        - 양수: 과신 (틀렸는데 확신함)
        - 음수: 신중 (맞았는데 확신 낮음)
        
        Returns:
            업데이트된 illusion_avg 값
        """
        from analytics.models import TestResult
        
        # 해당 사용자의 모든 테스트 결과에서 illusion_score 평균 계산
        result = TestResult.objects.filter(
            session__user=self
        ).aggregate(avg_illusion=Avg('illusion_score'))
        
        avg_value = result['avg_illusion']
        if avg_value is not None:
            self.illusion_avg = avg_value
            self.save(update_fields=['illusion_avg'])
        
        return self.illusion_avg
    
    @property
    def metacognition_tendency(self) -> str:
        """메타인지 성향 문자열 반환"""
        if self.illusion_avg > 0.1:
            return '과신 (Overconfident)'
        elif self.illusion_avg < -0.1:
            return '신중 (Underconfident)'
        else:
            return '균형 (Calibrated)'


# =============================================================================
# [4] 테스트 세션 모델 (Time-series axis)
# =============================================================================

class TestSession(models.Model):
    """
    실험/학습 세션 모델
    
    에빙하우스 망각 곡선에 기반한 시간 축:
    - T0: 학습 직후
    - T1: 20분 후
    - T2: 4시간 후
    - T3: 24시간 후
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='세션 ID',
        help_text='테스트 세션 고유 식별자 (UUID)'
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='test_sessions',
        verbose_name='사용자',
        help_text='테스트를 수행하는 사용자'
    )
    time_point = models.CharField(
        max_length=10,
        choices=TimePoint.choices,
        default=TimePoint.T0,
        verbose_name='측정 시점',
        help_text='망각 곡선 기반 측정 시점 (T0: 직후, T1: 20분, T2: 4시간, T3: 24시간)'
    )
    scheduled_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name='예정 시각',
        help_text='테스트 예정 시각'
    )
    performed_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name='수행 시각',
        help_text='테스트 실제 수행 시각'
    )

    class Meta:
        verbose_name = '테스트 세션'
        verbose_name_plural = '테스트 세션 목록'
        ordering = ['-scheduled_at']
        indexes = [
            models.Index(fields=['user', 'time_point']),
            models.Index(fields=['scheduled_at']),
            models.Index(fields=['performed_at']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.get_time_point_display()}"


# =============================================================================
# [5] 테스트 결과 모델 (Core Observation)
# =============================================================================

class TestResult(models.Model):
    """
    인지 측정 결과 모델 (핵심 관측 데이터)
    
    측정 지표:
    - is_correct (A): 정답 여부 (0 또는 1)
    - confidence_score (C): 주관적 확신도 (0.0 ~ 1.0)
    - illusion_score: C - A (메타인지 착각 지수)
    - response_time_ms (RT): 인출 반응 시간 (ms)
    """
    id = models.BigAutoField(
        primary_key=True,
        verbose_name='결과 ID',
        help_text='테스트 결과 고유 식별자'
    )
    session = models.ForeignKey(
        TestSession,
        on_delete=models.CASCADE,
        related_name='results',
        verbose_name='테스트 세션',
        help_text='이 결과가 속한 테스트 세션'
    )
    node = models.ForeignKey(
        'knowledge.KnowledgeNode',
        on_delete=models.CASCADE,
        related_name='test_results',
        verbose_name='지식 노드',
        help_text='테스트 대상 지식 노드'
    )
    
    # 메타인지 및 정확도
    is_correct = models.BooleanField(
        verbose_name='정답 여부',
        help_text='A: 정답 여부 (True=1, False=0)'
    )
    confidence_score = models.FloatField(
        verbose_name='확신도',
        help_text='C: 주관적 확신도 (0.0 ~ 1.0)'
    )
    illusion_score = models.FloatField(
        null=True,
        blank=True,
        verbose_name='착각 지수',
        help_text='C - A: 메타인지 착각 지수 (자동 계산됨)'
    )
    
    # 인출 노력
    response_time_ms = models.IntegerField(
        verbose_name='반응 시간',
        help_text='RT: 인출 반응 시간 (밀리초)'
    )
    test_type = models.CharField(
        max_length=20,
        choices=TestType.choices,
        verbose_name='테스트 유형',
        help_text='A1: 상향식, A2: 하향식(설명), B: 자유회상'
    )

    class Meta:
        verbose_name = '테스트 결과'
        verbose_name_plural = '테스트 결과 목록'
        ordering = ['-id']
        indexes = [
            models.Index(fields=['session', 'node']),
            models.Index(fields=['test_type']),
            models.Index(fields=['is_correct']),
        ]

    def __str__(self):
        correct_str = "✓" if self.is_correct else "✗"
        return f"{self.node.title} [{correct_str}] C={self.confidence_score:.2f}"
    
    def clean(self):
        """유효성 검사"""
        if self.confidence_score is not None:
            if not (0.0 <= self.confidence_score <= 1.0):
                raise ValidationError({
                    'confidence_score': '확신도는 0.0에서 1.0 사이여야 합니다.'
                })
        
        if self.response_time_ms is not None:
            if self.response_time_ms < 0:
                raise ValidationError({
                    'response_time_ms': '반응 시간은 0 이상이어야 합니다.'
                })
    
    def save(self, *args, **kwargs):
        """
        저장 시 illusion_score 자동 계산
        
        illusion_score = C - A
        - A (is_correct): 0 또는 1로 변환
        - C (confidence_score): 0.0 ~ 1.0
        
        결과 해석:
        - 양수: 과신 (틀렸는데 확신함)
        - 음수: 신중 (맞았는데 확신 낮음)
        - 0 근처: 잘 보정됨
        """
        # is_correct를 0/1 정수로 변환하여 계산
        accuracy_value = 1.0 if self.is_correct else 0.0
        self.illusion_score = self.confidence_score - accuracy_value
        
        self.full_clean()
        super().save(*args, **kwargs)


# =============================================================================
# [6] 설명형 심층 신호 분석 (A2 Descriptive Only)
# =============================================================================

class SpeechAnalysis(models.Model):
    """
    설명형 심층 신호 분석 모델 (A2 전용)
    
    ⚠️ 이 모델은 test_type이 'A2_TOP_DOWN'인 TestResult에만 연결되어야 합니다.
    설명형 테스트(A2)에서 사용자의 발화 패턴을 분석합니다.
    
    분석 지표:
    - pause_count: 500ms 이상 침묵 구간 수 (인출 어려움 지표)
    - total_pause_duration: 총 침묵 시간 (인지 부하 지표)
    - speech_segments: 발화 마디 수 (설명 세분화 정도)
    - text_length: 응답 텍스트 길이 (지식 풍부함 지표)
    """
    result = models.OneToOneField(
        TestResult,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='speech_analysis',
        verbose_name='테스트 결과',
        help_text='연결된 테스트 결과 (1:1 관계, A2 유형만)'
    )
    pause_count = models.IntegerField(
        default=0,
        verbose_name='침묵 구간 수',
        help_text='500ms 이상 침묵 구간 수 (인출 어려움 지표)'
    )
    total_pause_duration = models.IntegerField(
        default=0,
        verbose_name='총 침묵 시간',
        help_text='총 침묵 시간 (밀리초)'
    )
    speech_segments = models.IntegerField(
        default=0,
        verbose_name='발화 마디 수',
        help_text='발화 마디 수 (설명 세분화 정도)'
    )
    text_length = models.IntegerField(
        default=0,
        verbose_name='텍스트 길이',
        help_text='응답 텍스트 길이 (문자 수)'
    )

    class Meta:
        verbose_name = '발화 분석'
        verbose_name_plural = '발화 분석 목록'

    def __str__(self):
        return f"SpeechAnalysis(result={self.result_id}, pauses={self.pause_count})"
    
    def clean(self):
        """
        유효성 검사: A2_TOP_DOWN 유형만 허용
        
        ⚠️ SpeechAnalysis는 설명형 테스트(A2_TOP_DOWN)에만 생성됩니다.
        다른 test_type에서는 ValidationError가 발생합니다.
        """
        if self.result and self.result.test_type != TestType.A2_TOP_DOWN:
            raise ValidationError({
                'result': f"SpeechAnalysis는 A2_TOP_DOWN 테스트에만 생성 가능합니다. "
                          f"현재: {self.result.test_type}"
            })
    
    def save(self, *args, **kwargs):
        """저장 시 유효성 검사 수행"""
        self.full_clean()
        super().save(*args, **kwargs)
    
    @property
    def retrieval_difficulty_index(self) -> float:
        """
        인출 난이도 지수 계산
        
        = (pause_count * avg_pause_duration) / speech_segments
        침묵이 많고 길수록, 발화 마디가 적을수록 높은 값
        """
        if self.speech_segments == 0:
            return float('inf')
        
        avg_pause = self.total_pause_duration / max(self.pause_count, 1)
        return (self.pause_count * avg_pause) / self.speech_segments
