"""
Cognitive Benchmark Service - 초기 인지 실험 설계 및 실행

망각 곡선 추정을 위한 초기 벤치마크 테스트:
- 익숙한 도메인 (CS) vs 익숙하지 않은 도메인 (경상도 사투리)
- T0~T3 시간축 기반 기억 형성/붕괴 측정
- illusion_score, RT 기반 인지 특성 분석
"""

import math
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from django.db import models
from django.db.models import Avg, Count, Q
from django.utils import timezone

logger = logging.getLogger(__name__)


# =============================================================================
# 도메인 정의
# =============================================================================

class Domain(Enum):
    """테스트 도메인"""
    CS = "cs"                    # 익숙한 도메인 (Computer Science)
    DIALECT = "dialect"          # 익숙하지 않은 도메인 (경상도 사투리)


# CS 도메인 태그
CS_TAGS = [
    "데이터베이스", "운영체제", "자료구조", "네트워크", "알고리즘",
    "컴퓨터과학", "프로그래밍", "소프트웨어", "시스템",
    "database", "os", "data_structure", "network", "algorithm"
]

# 경상도 사투리 도메인 태그
DIALECT_TAGS = [
    "경상도", "사투리", "방언", "지역표현", "구어체",
    "경상도_사투리", "경상도_방언", "dialect"
]


# =============================================================================
# 결과 데이터 클래스
# =============================================================================

@dataclass
class TimePointStats:
    """시점별 통계"""
    time_point: str
    accuracy: float          # 정답률 (0.0 ~ 1.0)
    avg_rt_ms: float         # 평균 반응 시간 (ms)
    avg_confidence: float    # 평균 확신도
    avg_illusion: float      # 평균 착각 지수
    sample_count: int        # 샘플 수


@dataclass
class DomainAnalysis:
    """도메인별 분석 결과"""
    domain: str
    time_point_stats: List[TimePointStats]
    forgetting_k: float               # 망각 기울기
    encoding_strength: float          # 초기 기억 형성 강도
    retention_rate_t3: float          # T3 시점 유지율
    avg_illusion_score: float         # 평균 착각 지수
    illusion_tendency: str            # 과신/신중/균형


@dataclass
class BenchmarkResult:
    """벤치마크 전체 결과"""
    user_id: str
    cs_analysis: DomainAnalysis
    dialect_analysis: DomainAnalysis
    base_forgetting_k: float          # 추정된 기초 망각 상수
    domain_forgetting_ratio: float    # k_dialect / k_cs (도메인 차이)
    overall_illusion_avg: float       # 전체 평균 착각 지수
    recommendation: str               # 복습 스케줄 권장사항


# =============================================================================
# 복습 스케줄 계산
# =============================================================================

def calculate_next_review_hours(k: float, target_retention: float) -> float:
    """
    목표 암기율을 만족하는 다음 복습 시점 계산
    
    에빙하우스 망각 곡선 공식:
        R(t) = exp(-k * t)
    
    역산하여 t를 구함:
        t = -ln(R_target) / k
    
    Args:
        k: 망각 계수 (단위: 1/hour, k > 0)
        target_retention: 목표 암기율 (0 < R < 1)
        
    Returns:
        다음 복습까지의 시간 (hours)
        
    Raises:
        ValueError: k <= 0 또는 target_retention이 (0, 1) 범위 밖일 경우
        
    Examples:
        >>> calculate_next_review_hours(k=0.0213, target_retention=0.8)
        10.48  # 약 10.5시간 후 복습
        
        >>> calculate_next_review_hours(k=0.132, target_retention=0.8)
        1.69   # 약 1.7시간 후 복습
    """
    # 입력 검증
    if k <= 0:
        raise ValueError(f"망각 계수 k는 양수여야 합니다: k={k}")
    
    if not (0 < target_retention < 1):
        raise ValueError(
            f"목표 암기율은 0과 1 사이여야 합니다: target_retention={target_retention}"
        )
    
    # t = -ln(R_target) / k
    # ln(0.8) = -0.223, so t = 0.223 / k
    t_next = -math.log(target_retention) / k
    
    return t_next


@dataclass
class ReviewSchedule:
    """복습 스케줄 정보"""
    target_retention: float           # 목표 암기율
    cs_review_hours: float            # CS 도메인 복습 시점 (시간)
    dialect_review_hours: float       # 사투리 도메인 복습 시점 (시간)
    cs_review_datetime: Optional[datetime] = None    # CS 복습 시각
    dialect_review_datetime: Optional[datetime] = None  # 사투리 복습 시각


class ReviewScheduleCalculator:
    """
    복습 스케줄 계산기
    
    망각 곡선 기반으로 최적의 복습 시점을 계산합니다.
    """
    
    # 기본 목표 암기율
    DEFAULT_TARGET_RETENTION = 0.8
    
    # 암기율별 권장 복습 간격 임계값
    RETENTION_THRESHOLDS = {
        'high': 0.9,      # 높은 유지율 목표
        'medium': 0.8,    # 중간 유지율 목표 (기본)
        'low': 0.7,       # 낮은 유지율 허용
    }
    
    def calculate_review_schedule(
        self,
        k_cs: float,
        k_dialect: float,
        target_retention: float = None,
        from_time: datetime = None
    ) -> ReviewSchedule:
        """
        도메인별 복습 스케줄 계산
        
        Args:
            k_cs: CS 도메인 망각 계수
            k_dialect: 사투리 도메인 망각 계수
            target_retention: 목표 암기율 (기본: 0.8)
            from_time: 기준 시간 (기본: 현재)
            
        Returns:
            ReviewSchedule 객체
        """
        if target_retention is None:
            target_retention = self.DEFAULT_TARGET_RETENTION
        
        if from_time is None:
            from_time = timezone.now()
        
        # 각 도메인별 복습 시간 계산
        cs_hours = calculate_next_review_hours(k_cs, target_retention)
        dialect_hours = calculate_next_review_hours(k_dialect, target_retention)
        
        # 복습 시각 계산
        cs_datetime = from_time + timedelta(hours=cs_hours)
        dialect_datetime = from_time + timedelta(hours=dialect_hours)
        
        return ReviewSchedule(
            target_retention=target_retention,
            cs_review_hours=round(cs_hours, 2),
            dialect_review_hours=round(dialect_hours, 2),
            cs_review_datetime=cs_datetime,
            dialect_review_datetime=dialect_datetime
        )
    
    def calculate_multi_retention_schedules(
        self,
        k_cs: float,
        k_dialect: float
    ) -> Dict[str, ReviewSchedule]:
        """
        여러 목표 암기율에 대한 복습 스케줄 계산
        
        Returns:
            {'high': ReviewSchedule, 'medium': ReviewSchedule, 'low': ReviewSchedule}
        """
        return {
            level: self.calculate_review_schedule(k_cs, k_dialect, retention)
            for level, retention in self.RETENTION_THRESHOLDS.items()
        }
    
    def format_hours_to_human_readable(self, hours: float) -> str:
        """
        시간을 사람이 읽기 쉬운 형식으로 변환
        
        Examples:
            0.5 -> "30분"
            1.5 -> "1시간 30분"
            25.0 -> "1일 1시간"
        """
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes}분"
        elif hours < 24:
            h = int(hours)
            m = int((hours - h) * 60)
            if m > 0:
                return f"{h}시간 {m}분"
            return f"{h}시간"
        else:
            days = int(hours / 24)
            remaining_hours = int(hours % 24)
            if remaining_hours > 0:
                return f"{days}일 {remaining_hours}시간"
            return f"{days}일"


# =============================================================================
# 망각곡선 시각화 데이터 생성
# =============================================================================

# 기본 시간 샘플링 포인트 (시간 단위)
DEFAULT_CURVE_TIME_POINTS = [0, 1, 3, 6, 12, 24, 48]


def calculate_retention(k: float, t: float) -> float:
    """
    특정 시점의 기억 유지율 계산
    
    Ebbinghaus 망각 곡선: R(t) = exp(-k * t)
    
    Args:
        k: 망각 계수 (1/hour)
        t: 경과 시간 (hour)
        
    Returns:
        기억 유지율 (0.0 ~ 1.0)
    """
    return math.exp(-k * t)


def generate_forgetting_curve(
    k: float, 
    time_points: List[float] = None
) -> List[Dict[str, float]]:
    """
    망각곡선 시각화용 좌표 데이터 생성
    
    프론트엔드에서 직접 그래프를 그릴 수 있도록
    (t, retention) 좌표 배열을 생성합니다.
    
    Args:
        k: 망각 계수 (1/hour)
        time_points: 샘플링할 시간 포인트 리스트 (기본: [0,1,3,6,12,24,48])
        
    Returns:
        [{"t": 0, "retention": 1.0}, {"t": 6, "retention": 0.89}, ...]
        
    Example:
        >>> generate_forgetting_curve(k=0.01, time_points=[0, 6, 12, 24])
        [
            {"t": 0, "retention": 1.0},
            {"t": 6, "retention": 0.942},
            {"t": 12, "retention": 0.887},
            {"t": 24, "retention": 0.787}
        ]
    """
    if time_points is None:
        time_points = DEFAULT_CURVE_TIME_POINTS
    
    curve_data = []
    for t in time_points:
        retention = calculate_retention(k, t)
        curve_data.append({
            "t": t,
            "retention": round(retention, 3)
        })
    
    return curve_data


@dataclass
class ForgettingCurveData:
    """망각곡선 시각화 데이터"""
    cs: List[Dict[str, float]]
    dialect: List[Dict[str, float]]


class ForgettingCurveGenerator:
    """
    망각곡선 시각화 데이터 생성기
    
    프론트엔드에서 수식을 알 필요 없이
    바로 그래프를 렌더링할 수 있는 데이터를 제공합니다.
    """
    
    def __init__(self, time_points: List[float] = None):
        """
        Args:
            time_points: 커스텀 샘플링 시간 포인트
        """
        self.time_points = time_points or DEFAULT_CURVE_TIME_POINTS
    
    def generate(
        self, 
        k_cs: float, 
        k_dialect: float
    ) -> ForgettingCurveData:
        """
        도메인별 망각곡선 데이터 생성
        
        Args:
            k_cs: CS 도메인 망각 계수
            k_dialect: 사투리 도메인 망각 계수
            
        Returns:
            ForgettingCurveData 객체
        """
        return ForgettingCurveData(
            cs=generate_forgetting_curve(k_cs, self.time_points),
            dialect=generate_forgetting_curve(k_dialect, self.time_points)
        )
    
    def to_dict(
        self, 
        k_cs: float, 
        k_dialect: float
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        JSON 직렬화 가능한 딕셔너리 반환
        
        Returns:
            {
                "cs": [{"t": 0, "retention": 1.0}, ...],
                "dialect": [{"t": 0, "retention": 1.0}, ...]
            }
        """
        data = self.generate(k_cs, k_dialect)
        return {
            "cs": data.cs,
            "dialect": data.dialect
        }


# =============================================================================
# 노드 선정 서비스
# =============================================================================

class NodeSelector:
    """
    테스트 대상 노드 선정
    
    각 도메인에서 적절한 노드를 선택하여 테스트 세트 구성
    
    분류 기준:
    - CS (TRACK_A): Computer Science 관련 노드, 방언/사투리 태그 제외
    - Dialect (TRACK_B): 경상도/전라도 사투리, 방언 관련 노드
    """
    
    # 방언/사투리 관련 태그 (CS에서 제외해야 할 태그)
    DIALECT_EXCLUDE_TAGS = [
        '경상도', '전라도', '사투리', '방언', '지역표현', '구어체',
        '경상도_사투리', '경상도_방언', '전라도_사투리', '전라도_방언',
        'dialect', '향토문화', '지역문화', '민속',
    ]
    
    # CS 관련 태그 (순수 CS 노드 식별용)
    CS_INCLUDE_TAGS = [
        '데이터베이스', '운영체제', '자료구조', '네트워크', '알고리즘',
        '컴퓨터과학', '프로그래밍', '소프트웨어', '시스템',
        'database', 'os', 'data_structure', 'network', 'algorithm',
        'computer_science', 'programming', 'software', 'machine_learning',
        '머신러닝', '딥러닝', 'deep_learning', 'api', 'backend', 'frontend',
    ]
    
    def __init__(self, nodes_per_domain: int = 10):
        """
        Args:
            nodes_per_domain: 도메인당 선택할 노드 수 (권장: 8~12)
        """
        self.nodes_per_domain = nodes_per_domain
    
    def _has_dialect_tags(self, node) -> bool:
        """노드가 방언/사투리 관련 태그를 가지고 있는지 확인"""
        if not node.tags:
            return False
        
        node_tags_lower = [str(tag).lower() for tag in node.tags]
        for exclude_tag in self.DIALECT_EXCLUDE_TAGS:
            for node_tag in node_tags_lower:
                if exclude_tag.lower() in node_tag:
                    return True
        return False
    
    def _has_cs_tags(self, node) -> bool:
        """노드가 CS 관련 태그를 가지고 있는지 확인"""
        if not node.tags:
            return False
        
        node_tags_lower = [str(tag).lower() for tag in node.tags]
        for cs_tag in self.CS_INCLUDE_TAGS:
            for node_tag in node_tags_lower:
                if cs_tag.lower() in node_tag:
                    return True
        return False
    
    def select_cs_nodes(self) -> List:
        """
        CS 도메인 노드 선정
        
        선정 기준 (우선순위):
        1. TrackType.TRACK_A + CS 관련 태그 노드 (최우선)
        2. TrackType.TRACK_A + 방언 태그 없는 노드 (보조)
        3. 중립 노드 (TRACK_B 및 방언 태그 제외)
        """
        from knowledge.models import KnowledgeNode, TrackType
        import random
        
        # 1단계: TRACK_A 노드 중 CS 태그가 있고 방언 태그가 없는 노드 (최우선)
        all_track_a = list(KnowledgeNode.objects.filter(
            track_type=TrackType.TRACK_A
        ))
        
        # Tier 1: CS 태그 있음 + 방언 태그 없음
        tier1_candidates = [
            node for node in all_track_a 
            if self._has_cs_tags(node) and not self._has_dialect_tags(node)
        ]
        
        # Tier 2: CS 태그 없지만 방언 태그도 없음 (중립 노드)
        tier2_candidates = [
            node for node in all_track_a 
            if not self._has_cs_tags(node) and not self._has_dialect_tags(node)
        ]
        
        # 로깅
        logger.info(f"[NodeSelector] TRACK_A 분석: 총 {len(all_track_a)}개, "
                   f"CS태그 {len(tier1_candidates)}개, 중립 {len(tier2_candidates)}개")
        
        # Tier 1에서 우선 선정
        random.shuffle(tier1_candidates)
        selected = tier1_candidates[:self.nodes_per_domain]
        
        # Tier 1이 부족하면 Tier 2에서 추가
        if len(selected) < self.nodes_per_domain:
            needed = self.nodes_per_domain - len(selected)
            random.shuffle(tier2_candidates)
            selected.extend(tier2_candidates[:needed])
        
        # 2단계: 여전히 부족하면 다른 노드에서 추가 (TRACK_B 및 방언 태그 제외)
        if len(selected) < self.nodes_per_domain:
            existing_ids = {node.id for node in selected}
            
            additional_candidates = [
                node for node in KnowledgeNode.objects.exclude(
                    id__in=existing_ids
                ).exclude(
                    track_type=TrackType.TRACK_B
                )
                if not self._has_dialect_tags(node)
            ]
            
            random.shuffle(additional_candidates)
            needed = self.nodes_per_domain - len(selected)
            selected.extend(additional_candidates[:needed])
        
        logger.info(f"[NodeSelector] CS 도메인 노드 {len(selected)}개 선정 (TRACK_A 기반, 방언 태그 제외)")
        return selected
    
    def select_dialect_nodes(self) -> List:
        """
        경상도/전라도 사투리 도메인 노드 선정
        
        선정 기준:
        1. TrackType.TRACK_B인 노드 우선
        2. 부족하면 방언/사투리 관련 태그가 있는 TRACK_A 노드도 포함
        """
        from knowledge.models import KnowledgeNode, TrackType
        import random
        
        # 1단계: TRACK_B 노드 선정
        track_b_nodes = list(KnowledgeNode.objects.filter(
            track_type=TrackType.TRACK_B
        ))
        random.shuffle(track_b_nodes)
        selected = track_b_nodes[:self.nodes_per_domain]
        
        # 2단계: 부족하면 TRACK_A 중 방언 태그가 있는 노드 추가
        if len(selected) < self.nodes_per_domain:
            existing_ids = {node.id for node in selected}
            
            dialect_tagged_nodes = [
                node for node in KnowledgeNode.objects.filter(
                    track_type=TrackType.TRACK_A
                ).exclude(id__in=existing_ids)
                if self._has_dialect_tags(node)
            ]
            
            random.shuffle(dialect_tagged_nodes)
            needed = self.nodes_per_domain - len(selected)
            selected.extend(dialect_tagged_nodes[:needed])
            
            if dialect_tagged_nodes:
                logger.info(f"[NodeSelector] TRACK_A에서 방언 태그 노드 {min(needed, len(dialect_tagged_nodes))}개 추가 선정")
        
        logger.info(f"[NodeSelector] 사투리 도메인 노드 {len(selected)}개 선정 (TRACK_B + 방언 태그)")
        return selected
    
    def get_test_nodes(self) -> Dict[str, List]:
        """테스트용 전체 노드 세트 반환"""
        cs_nodes = self.select_cs_nodes()
        dialect_nodes = self.select_dialect_nodes()
        
        # 중복 검증: 같은 노드가 양쪽에 포함되어 있지 않은지 확인
        cs_ids = {node.id for node in cs_nodes}
        dialect_ids = {node.id for node in dialect_nodes}
        overlap = cs_ids & dialect_ids
        
        if overlap:
            logger.warning(f"[NodeSelector] CS/Dialect 중복 노드 {len(overlap)}개 발견, Dialect에서 제거")
            dialect_nodes = [node for node in dialect_nodes if node.id not in overlap]
        
        return {
            Domain.CS.value: cs_nodes,
            Domain.DIALECT.value: dialect_nodes
        }


# =============================================================================
# 테스트 세션 생성기
# =============================================================================

class SessionScheduler:
    """
    테스트 세션 스케줄 생성
    
    에빙하우스 망각 곡선 기반 시간 간격:
    - T0: 즉시 (초기 인지 테스트)
    - T1: 10분 후
    - T2: 1시간 후
    - T3: 24시간 후
    """
    
    # 시간 간격 정의 (분 단위)
    TIME_INTERVALS = {
        'T0': 0,
        'T1': 10,      # 10분
        'T2': 60,      # 1시간
        'T3': 1440,    # 24시간
    }
    
    def create_sessions(self, user, start_time: datetime = None) -> Dict:
        """
        사용자의 테스트 세션 생성
        
        Args:
            user: User 모델 인스턴스
            start_time: 시작 시간 (기본: 현재)
            
        Returns:
            시점별 세션 딕셔너리
        """
        from analytics.models import TestSession, TimePoint
        
        if start_time is None:
            start_time = timezone.now()
        
        sessions = {}
        
        for time_point, minutes in self.TIME_INTERVALS.items():
            scheduled_at = start_time + timedelta(minutes=minutes)
            
            session = TestSession.objects.create(
                user=user,
                time_point=time_point,
                scheduled_at=scheduled_at
            )
            sessions[time_point] = session
            
            logger.info(f"[SessionScheduler] 세션 생성: {time_point} @ {scheduled_at}")
        
        return sessions


# =============================================================================
# 테스트 결과 기록기
# =============================================================================

class ResultRecorder:
    """
    테스트 결과 기록
    
    각 테스트 결과를 저장하고 발화 분석 메타데이터를 관리
    """
    
    def record_result(
        self,
        session,
        node,
        is_correct: bool,
        confidence_score: float,
        response_time_ms: int,
        test_type: str,
        speech_data: Optional[Dict] = None
    ):
        """
        테스트 결과 기록
        
        Args:
            session: TestSession 인스턴스
            node: KnowledgeNode 인스턴스
            is_correct: 정답 여부
            confidence_score: 확신도 (0~1)
            response_time_ms: 반응 시간 (ms)
            test_type: 테스트 유형 (A1_BOTTOM_UP, A2_TOP_DOWN, B_RECALL)
            speech_data: A2 테스트 시 발화 분석 데이터
        """
        from analytics.models import TestResult, SpeechAnalysis, TestType
        
        # 테스트 결과 생성 (illusion_score는 자동 계산됨)
        result = TestResult.objects.create(
            session=session,
            node=node,
            is_correct=is_correct,
            confidence_score=confidence_score,
            response_time_ms=response_time_ms,
            test_type=test_type
        )
        
        # A2 테스트의 경우 발화 분석 저장
        if test_type == TestType.A2_TOP_DOWN and speech_data:
            SpeechAnalysis.objects.create(
                result=result,
                pause_count=speech_data.get('pause_count', 0),
                total_pause_duration=speech_data.get('total_pause_duration', 0),
                speech_segments=speech_data.get('speech_segments', 0),
                text_length=speech_data.get('text_length', 0)
            )
        
        # 노드의 안정성/난이도 지수 업데이트
        node.update_stability_index(response_time_ms, is_correct)
        node.update_difficulty_index(response_time_ms, is_correct)
        
        logger.debug(f"[ResultRecorder] 결과 기록: {node.title} @ {session.time_point}")
        
        return result


# =============================================================================
# 망각 곡선 분석기
# =============================================================================

class ForgettingCurveAnalyzer:
    """
    망각 곡선 분석
    
    시간 경과에 따른 정답률 변화를 분석하여 망각 기울기 k 추정
    에빙하우스 망각 곡선: R(t) = e^(-t/S) 또는 R(t) = e^(-k*t)
    """
    
    # 시간 간격 (분 → 시간)
    TIME_POINTS_HOURS = {
        'T0': 0,
        'T1': 10/60,    # ~0.167시간
        'T2': 1,        # 1시간
        'T3': 24,       # 24시간
    }
    
    def calculate_forgetting_k(self, time_point_stats: List[TimePointStats]) -> float:
        """
        망각 기울기 k 계산
        
        선형 회귀를 사용하여 log(R) = -k*t + log(R0) 형태로 추정
        R: 유지율 (정답률)
        t: 시간 (시간 단위)
        
        Args:
            time_point_stats: 시점별 통계 리스트
            
        Returns:
            망각 기울기 k (클수록 빠르게 망각)
        """
        if len(time_point_stats) < 2:
            return 0.5  # 기본값
        
        # 데이터 포인트 추출
        points = []
        for stat in time_point_stats:
            t = self.TIME_POINTS_HOURS.get(stat.time_point, 0)
            r = max(stat.accuracy, 0.01)  # log(0) 방지
            points.append((t, math.log(r)))
        
        # 선형 회귀 (최소제곱법)
        n = len(points)
        sum_t = sum(p[0] for p in points)
        sum_log_r = sum(p[1] for p in points)
        sum_t2 = sum(p[0]**2 for p in points)
        sum_t_log_r = sum(p[0] * p[1] for p in points)
        
        denominator = n * sum_t2 - sum_t**2
        if abs(denominator) < 1e-10:
            return 0.5  # 기본값
        
        # k = -slope
        slope = (n * sum_t_log_r - sum_t * sum_log_r) / denominator
        k = -slope
        
        # 음수 방지 및 범위 제한 (0.01 ~ 5.0)
        k = max(0.01, min(k, 5.0))
        
        logger.info(f"[ForgettingCurve] k = {k:.4f}")
        return k
    
    def calculate_encoding_strength(self, t0_stats: TimePointStats) -> float:
        """
        초기 기억 형성 강도 계산
        
        T0 시점의 정답률과 RT를 기반으로 인코딩 강도 추정
        
        Args:
            t0_stats: T0 시점 통계
            
        Returns:
            인코딩 강도 (0~1)
        """
        # 정답률 기여
        accuracy_factor = t0_stats.accuracy
        
        # RT 기여 (빠를수록 강함, 5초를 기준으로 정규화)
        rt_factor = 1 - min(t0_stats.avg_rt_ms / 5000, 1.0)
        
        # 확신도 기여
        confidence_factor = t0_stats.avg_confidence
        
        # 가중 평균
        encoding_strength = (
            0.5 * accuracy_factor + 
            0.3 * rt_factor + 
            0.2 * confidence_factor
        )
        
        return encoding_strength
    
    def analyze_domain(
        self, 
        user_id: str, 
        domain: Domain, 
        node_ids: List[str]
    ) -> DomainAnalysis:
        """
        도메인별 분석 수행
        
        Args:
            user_id: 사용자 ID
            domain: 도메인 유형
            node_ids: 해당 도메인의 노드 ID 리스트
            
        Returns:
            DomainAnalysis 결과
        """
        from analytics.models import TestResult, TestSession, TimePoint
        
        time_point_stats = []
        
        for tp in ['T0', 'T1', 'T2', 'T3']:
            # 해당 시점의 결과 조회
            results = TestResult.objects.filter(
                session__user_id=user_id,
                session__time_point=tp,
                node_id__in=node_ids
            )
            
            if not results.exists():
                continue
            
            # 통계 계산
            stats = results.aggregate(
                accuracy=Avg('is_correct', output_field=models.FloatField()),
                avg_rt=Avg('response_time_ms'),
                avg_conf=Avg('confidence_score'),
                avg_illusion=Avg('illusion_score')
            )
            
            time_point_stats.append(TimePointStats(
                time_point=tp,
                accuracy=float(stats['accuracy'] or 0),
                avg_rt_ms=float(stats['avg_rt'] or 0),
                avg_confidence=float(stats['avg_conf'] or 0),
                avg_illusion=float(stats['avg_illusion'] or 0),
                sample_count=results.count()
            ))
        
        # 망각 기울기 계산
        forgetting_k = self.calculate_forgetting_k(time_point_stats)
        
        # 초기 인코딩 강도
        t0_stats = next((s for s in time_point_stats if s.time_point == 'T0'), None)
        encoding_strength = self.calculate_encoding_strength(t0_stats) if t0_stats else 0.5
        
        # T3 유지율
        t3_stats = next((s for s in time_point_stats if s.time_point == 'T3'), None)
        retention_rate_t3 = t3_stats.accuracy if t3_stats else 0
        
        # 평균 착각 지수
        all_illusion = [s.avg_illusion for s in time_point_stats if s.avg_illusion is not None]
        avg_illusion = sum(all_illusion) / len(all_illusion) if all_illusion else 0
        
        # 착각 성향 판정
        if avg_illusion > 0.1:
            illusion_tendency = "과신 (Overconfident)"
        elif avg_illusion < -0.1:
            illusion_tendency = "신중 (Underconfident)"
        else:
            illusion_tendency = "균형 (Calibrated)"
        
        return DomainAnalysis(
            domain=domain.value,
            time_point_stats=time_point_stats,
            forgetting_k=forgetting_k,
            encoding_strength=encoding_strength,
            retention_rate_t3=retention_rate_t3,
            avg_illusion_score=avg_illusion,
            illusion_tendency=illusion_tendency
        )


# =============================================================================
# 벤치마크 실행기
# =============================================================================

class CognitiveBenchmark:
    """
    초기 인지 벤치마크 테스트 실행기
    
    망각 곡선 추정을 위한 전체 테스트 파이프라인:
    1. 노드 선정
    2. 세션 스케줄링
    3. 결과 기록
    4. 분석 및 망각 상수 추정
    """
    
    def __init__(self, nodes_per_domain: int = 10):
        self.node_selector = NodeSelector(nodes_per_domain)
        self.session_scheduler = SessionScheduler()
        self.result_recorder = ResultRecorder()
        self.analyzer = ForgettingCurveAnalyzer()
    
    def initialize_benchmark(self, user) -> Dict:
        """
        벤치마크 초기화
        
        테스트 세션과 노드 세트를 생성
        
        Args:
            user: User 인스턴스
            
        Returns:
            초기화 정보 딕셔너리
        """
        # 노드 선정
        test_nodes = self.node_selector.get_test_nodes()
        
        # 세션 생성
        sessions = self.session_scheduler.create_sessions(user)
        
        logger.info(f"[CognitiveBenchmark] 초기화 완료: {user.username}")
        
        return {
            'user_id': str(user.id),
            'sessions': sessions,
            'nodes': test_nodes,
            'cs_node_ids': [str(n.id) for n in test_nodes[Domain.CS.value]],
            'dialect_node_ids': [str(n.id) for n in test_nodes[Domain.DIALECT.value]]
        }
    
    def submit_result(
        self,
        session,
        node,
        is_correct: bool,
        confidence_score: float,
        response_time_ms: int,
        test_type: str,
        speech_data: Optional[Dict] = None
    ):
        """테스트 결과 제출"""
        return self.result_recorder.record_result(
            session=session,
            node=node,
            is_correct=is_correct,
            confidence_score=confidence_score,
            response_time_ms=response_time_ms,
            test_type=test_type,
            speech_data=speech_data
        )
    
    def analyze_results(
        self,
        user_id: str,
        cs_node_ids: List[str],
        dialect_node_ids: List[str]
    ) -> BenchmarkResult:
        """
        벤치마크 결과 분석
        
        Args:
            user_id: 사용자 ID
            cs_node_ids: CS 도메인 노드 ID 리스트
            dialect_node_ids: 사투리 도메인 노드 ID 리스트
            
        Returns:
            BenchmarkResult
        """
        from analytics.models import User
        
        # 도메인별 분석
        cs_analysis = self.analyzer.analyze_domain(
            user_id, Domain.CS, cs_node_ids
        )
        dialect_analysis = self.analyzer.analyze_domain(
            user_id, Domain.DIALECT, dialect_node_ids
        )
        
        # 기초 망각 상수 추정 (두 도메인의 가중 평균)
        # 익숙한 도메인에 더 높은 가중치 부여
        base_k = (0.6 * cs_analysis.forgetting_k + 
                  0.4 * dialect_analysis.forgetting_k)
        
        # 도메인 간 망각 비율
        domain_ratio = (dialect_analysis.forgetting_k / cs_analysis.forgetting_k 
                        if cs_analysis.forgetting_k > 0 else 1.0)
        
        # 전체 평균 착각 지수
        overall_illusion = (cs_analysis.avg_illusion_score + 
                           dialect_analysis.avg_illusion_score) / 2
        
        # 복습 스케줄 권장사항 생성
        recommendation = self._generate_recommendation(
            base_k, domain_ratio, overall_illusion
        )
        
        # 사용자 모델 업데이트
        user = User.objects.get(id=user_id)
        user.base_forgetting_k = base_k
        user.update_illusion_avg()
        user.save(update_fields=['base_forgetting_k'])
        
        return BenchmarkResult(
            user_id=user_id,
            cs_analysis=cs_analysis,
            dialect_analysis=dialect_analysis,
            base_forgetting_k=base_k,
            domain_forgetting_ratio=domain_ratio,
            overall_illusion_avg=overall_illusion,
            recommendation=recommendation
        )
    
    def _generate_recommendation(
        self,
        base_k: float,
        domain_ratio: float,
        illusion_avg: float
    ) -> str:
        """복습 스케줄 권장사항 생성"""
        recommendations = []
        
        # 망각 속도 기반 권장
        if base_k > 1.0:
            recommendations.append("⚡ 빠른 망각 패턴: 짧은 간격의 반복 복습 권장 (10분, 1시간, 1일)")
        elif base_k > 0.3:
            recommendations.append("📊 보통 망각 패턴: 표준 간격 복습 권장 (1시간, 1일, 1주)")
        else:
            recommendations.append("🧠 느린 망각 패턴: 넉넉한 간격 복습 가능 (1일, 1주, 1개월)")
        
        # 도메인 차이 기반 권장
        if domain_ratio > 1.5:
            recommendations.append("📚 신규 도메인 학습 시 더 집중적인 복습 필요")
        elif domain_ratio < 0.8:
            recommendations.append("✨ 신규 도메인도 효과적으로 기억 유지 중")
        
        # 메타인지 기반 권장
        if illusion_avg > 0.15:
            recommendations.append("⚠️ 과신 경향: 자가 테스트 강화 및 확신도 재조정 필요")
        elif illusion_avg < -0.15:
            recommendations.append("💪 신중한 경향: 자신감 있게 복습 간격 확장 가능")
        
        return " | ".join(recommendations)


# =============================================================================
# 결과 리포터
# =============================================================================

class BenchmarkReporter:
    """
    벤치마크 결과 리포트 생성
    
    인지 특성 분석 관점의 서술적 리포트
    """
    
    def generate_report(
        self, 
        result: BenchmarkResult, 
        target_retention: float = 0.8
    ) -> Dict:
        """
        분석 리포트 생성
        
        프론트엔드에서 직접 사용할 수 있는 완전한 데이터 제공:
        - forgetting_curve: 망각곡선 그래프용 좌표 데이터
        - recommended_review: 복습 타이밍 시각화 정보
        
        Args:
            result: BenchmarkResult
            target_retention: 목표 암기율 (기본: 0.8)
            
        Returns:
            리포트 딕셔너리
        """
        k_cs = result.cs_analysis.forgetting_k
        k_dialect = result.dialect_analysis.forgetting_k
        
        # 복습 스케줄 계산
        schedule_calculator = ReviewScheduleCalculator()
        review_schedule = schedule_calculator.calculate_review_schedule(
            k_cs=k_cs,
            k_dialect=k_dialect,
            target_retention=target_retention
        )
        
        # 망각곡선 데이터 생성
        curve_generator = ForgettingCurveGenerator()
        forgetting_curve = curve_generator.to_dict(k_cs, k_dialect)
        
        return {
            'summary': {
                'user_id': result.user_id,
                'base_forgetting_k': round(result.base_forgetting_k, 4),
                'k_cs': round(k_cs, 4),
                'k_dialect': round(k_dialect, 4),
                'domain_ratio': round(result.domain_forgetting_ratio, 2),
                'overall_illusion': round(result.overall_illusion_avg, 3),
                'recommendation': result.recommendation
            },
            
            # =========================================================
            # 망각곡선 시각화 데이터 (프론트엔드 그래프용)
            # =========================================================
            'forgetting_curve': forgetting_curve,
            
            # =========================================================
            # 복습 타이밍 시각화 정보
            # =========================================================
            'recommended_review': {
                'target_retention': target_retention,
                'cs': {
                    'hours': review_schedule.cs_review_hours,
                    'label': schedule_calculator.format_hours_to_human_readable(
                        review_schedule.cs_review_hours
                    ),
                    'curve_x': review_schedule.cs_review_hours  # 그래프 x좌표
                },
                'dialect': {
                    'hours': review_schedule.dialect_review_hours,
                    'label': schedule_calculator.format_hours_to_human_readable(
                        review_schedule.dialect_review_hours
                    ),
                    'curve_x': review_schedule.dialect_review_hours  # 그래프 x좌표
                }
            },
            
            # =========================================================
            # 도메인별 상세 분석
            # =========================================================
            'cs_domain': self._format_domain_report(result.cs_analysis, "Computer Science"),
            'dialect_domain': self._format_domain_report(result.dialect_analysis, "경상도 사투리"),
            'temporal_comparison': self._format_temporal_comparison(
                result.cs_analysis, result.dialect_analysis
            ),
            'cognitive_interpretation': self._generate_interpretation(result)
        }
    
    def _format_domain_report(self, analysis: DomainAnalysis, domain_name: str) -> Dict:
        """도메인별 리포트 포맷"""
        return {
            'domain': domain_name,
            'forgetting_k': round(analysis.forgetting_k, 4),
            'encoding_strength': round(analysis.encoding_strength, 3),
            'retention_rate_24h': round(analysis.retention_rate_t3, 3),
            'illusion_tendency': analysis.illusion_tendency,
            'time_series': [
                {
                    'time_point': s.time_point,
                    'accuracy': round(s.accuracy, 3),
                    'avg_rt_ms': round(s.avg_rt_ms, 1),
                    'avg_confidence': round(s.avg_confidence, 3),
                    'illusion_score': round(s.avg_illusion, 3),
                    'sample_count': s.sample_count
                }
                for s in analysis.time_point_stats
            ]
        }
    
    def _format_temporal_comparison(
        self, 
        cs: DomainAnalysis, 
        dialect: DomainAnalysis
    ) -> Dict:
        """시간축 비교 데이터"""
        comparison = {}
        
        for tp in ['T0', 'T1', 'T2', 'T3']:
            cs_stat = next((s for s in cs.time_point_stats if s.time_point == tp), None)
            dialect_stat = next((s for s in dialect.time_point_stats if s.time_point == tp), None)
            
            comparison[tp] = {
                'cs_accuracy': round(cs_stat.accuracy, 3) if cs_stat else None,
                'dialect_accuracy': round(dialect_stat.accuracy, 3) if dialect_stat else None,
                'accuracy_difference': (
                    round(cs_stat.accuracy - dialect_stat.accuracy, 3)
                    if cs_stat and dialect_stat else None
                ),
                'cs_rt_ms': round(cs_stat.avg_rt_ms, 1) if cs_stat else None,
                'dialect_rt_ms': round(dialect_stat.avg_rt_ms, 1) if dialect_stat else None,
            }
        
        return comparison
    
    def _generate_interpretation(self, result: BenchmarkResult) -> Dict:
        """인지 특성 해석"""
        cs = result.cs_analysis
        dialect = result.dialect_analysis
        
        return {
            'forgetting_pattern': self._interpret_forgetting(cs, dialect),
            'encoding_pattern': self._interpret_encoding(cs, dialect),
            'metacognition_pattern': self._interpret_metacognition(cs, dialect),
            'domain_transfer': self._interpret_domain_transfer(result.domain_forgetting_ratio)
        }
    
    def _interpret_forgetting(self, cs: DomainAnalysis, dialect: DomainAnalysis) -> str:
        """망각 패턴 해석"""
        k_diff = dialect.forgetting_k - cs.forgetting_k
        
        if k_diff > 0.3:
            return (
                f"CS 도메인(k={cs.forgetting_k:.3f})에 비해 "
                f"사투리 도메인(k={dialect.forgetting_k:.3f})의 망각이 "
                f"{k_diff:.3f}만큼 빠름. 이는 사전 지식이 없는 새로운 정보가 "
                f"더 빠르게 붕괴되는 인지적 특성을 반영함."
            )
        elif k_diff < -0.1:
            return (
                f"흥미롭게도 사투리 도메인(k={dialect.forgetting_k:.3f})이 "
                f"CS 도메인(k={cs.forgetting_k:.3f})보다 기억 유지율이 높음. "
                f"이는 정서적 연결 또는 일화적 기억 효과일 수 있음."
            )
        else:
            return (
                f"두 도메인의 망각 기울기가 유사함 "
                f"(CS: {cs.forgetting_k:.3f}, 사투리: {dialect.forgetting_k:.3f}). "
                f"도메인 친숙도와 무관하게 일정한 기억 패턴을 보임."
            )
    
    def _interpret_encoding(self, cs: DomainAnalysis, dialect: DomainAnalysis) -> str:
        """인코딩 패턴 해석"""
        enc_diff = cs.encoding_strength - dialect.encoding_strength
        
        if enc_diff > 0.15:
            return (
                f"CS 도메인의 초기 인코딩 강도({cs.encoding_strength:.3f})가 "
                f"사투리 도메인({dialect.encoding_strength:.3f})보다 높음. "
                f"기존 스키마와의 연결이 초기 기억 형성을 강화함."
            )
        else:
            return (
                f"두 도메인의 초기 인코딩 강도가 유사함 "
                f"(CS: {cs.encoding_strength:.3f}, 사투리: {dialect.encoding_strength:.3f}). "
                f"학습 시점의 주의 집중도가 일정함을 시사함."
            )
    
    def _interpret_metacognition(self, cs: DomainAnalysis, dialect: DomainAnalysis) -> str:
        """메타인지 패턴 해석"""
        cs_ill = cs.avg_illusion_score
        dialect_ill = dialect.avg_illusion_score
        
        interpretations = []
        
        if cs_ill > 0.1:
            interpretations.append(
                f"CS 도메인에서 과신 경향(illusion={cs_ill:.2f}): "
                f"익숙한 도메인에서 자신의 기억을 과대평가하는 경향이 있음."
            )
        elif cs_ill < -0.1:
            interpretations.append(
                f"CS 도메인에서 신중한 경향(illusion={cs_ill:.2f}): "
                f"실제 정답률보다 확신도가 낮아 보수적 판단을 함."
            )
        
        if dialect_ill > 0.1:
            interpretations.append(
                f"사투리 도메인에서 과신 경향(illusion={dialect_ill:.2f}): "
                f"새로운 정보에 대해서도 자신감이 높음."
            )
        elif dialect_ill < -0.1:
            interpretations.append(
                f"사투리 도메인에서 신중한 경향(illusion={dialect_ill:.2f}): "
                f"불확실한 정보에 대해 적절히 낮은 확신도를 보임."
            )
        
        if not interpretations:
            interpretations.append(
                "두 도메인 모두에서 균형 잡힌 메타인지를 보임. "
                "확신도와 실제 정답률이 잘 일치함."
            )
        
        return " ".join(interpretations)
    
    def _interpret_domain_transfer(self, ratio: float) -> str:
        """도메인 전이 해석"""
        if ratio > 1.5:
            return (
                f"도메인 망각 비율이 {ratio:.2f}로, 신규 도메인 학습 시 "
                f"기존 도메인보다 약 {ratio:.1f}배 빠르게 망각됨. "
                f"새로운 도메인 학습 시 강화된 복습 스케줄 적용 권장."
            )
        elif ratio < 0.8:
            return (
                f"도메인 망각 비율이 {ratio:.2f}로, 신규 도메인에서도 "
                f"효과적인 기억 유지를 보임. 다양한 도메인 학습에 적합한 인지 특성."
            )
        else:
            return (
                f"도메인 망각 비율이 {ratio:.2f}로 균형적. "
                f"도메인 친숙도에 따른 망각 차이가 크지 않음."
            )
