"""
Track Type Classifier - 노드 트랙 타입 분류기

노드의 태그와 제목을 기반으로 TRACK_A (CS) 또는 TRACK_B (Dialect)를 자동 분류합니다.

사용법:
    from services.knowledge.track_classifier import classify_track_type
    from knowledge.models import TrackType
    
    track = classify_track_type(title="경상도 사투리", tags=["방언", "지역문화"])
    # 결과: TrackType.TRACK_B
    
    track = classify_track_type(title="데이터베이스", tags=["SQL", "DBMS"])
    # 결과: TrackType.TRACK_A
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 분류 기준 태그 정의
# =============================================================================

# Dialect (TRACK_B) 도메인 태그 - 방언, 사투리, 지역 문화 관련
DIALECT_TAGS = [
    # 지역
    '경상도', '전라도', '충청도', '강원도', '제주도',
    '경북', '경남', '전북', '전남',
    # 방언/사투리
    '사투리', '방언', '지역표현', '구어체', 'dialect',
    '경상도_사투리', '경상도_방언', '전라도_사투리', '전라도_방언',
    # 지역 문화
    '향토문화', '지역문화', '민속', '전통문화',
    # 기타 관련
    '억양', '발음', '음운', '어휘',
]

# CS (TRACK_A) 도메인 태그 - 컴퓨터 과학 관련
CS_TAGS = [
    # 데이터베이스
    '데이터베이스', 'database', 'db', 'sql', 'nosql', 'dbms',
    '관계형', '테이블', '쿼리', 'query', '스키마', 'schema',
    # 알고리즘/자료구조
    '알고리즘', 'algorithm', '자료구조', 'data_structure',
    '정렬', '탐색', '그래프', '트리', '해시',
    # 운영체제/시스템
    '운영체제', 'os', 'operating_system', '프로세스', '스레드', '메모리',
    # 네트워크
    '네트워크', 'network', 'tcp', 'ip', 'http', 'protocol',
    # 프로그래밍
    '프로그래밍', 'programming', '코딩', 'coding',
    '객체지향', 'oop', '함수형', 'functional',
    '파이썬', 'python', '자바', 'java', '자바스크립트', 'javascript',
    # 머신러닝/AI
    '머신러닝', 'machine_learning', 'ml',
    '딥러닝', 'deep_learning', 'dl',
    '인공지능', 'ai', 'artificial_intelligence',
    '신경망', 'neural_network', 'nn',
    # 소프트웨어 공학
    '소프트웨어', 'software', 'api', 'backend', 'frontend',
    '아키텍처', 'architecture', '디자인패턴', 'design_pattern',
    # 컴퓨터 과학 일반
    '컴퓨터과학', 'computer_science', 'cs',
]


def _normalize_tag(tag: str) -> str:
    """태그 정규화 (소문자, 공백 제거)"""
    return str(tag).lower().strip().replace(' ', '_')


def _has_matching_tags(tags: List[str], target_tags: List[str]) -> bool:
    """태그 리스트에서 대상 태그와 일치하는 항목이 있는지 확인"""
    if not tags:
        return False
    
    normalized_tags = [_normalize_tag(tag) for tag in tags]
    normalized_targets = [_normalize_tag(t) for t in target_tags]
    
    for node_tag in normalized_tags:
        for target in normalized_targets:
            # 부분 일치 (예: "경상도_사투리"에 "경상도" 포함)
            if target in node_tag or node_tag in target:
                return True
    
    return False


def _title_contains_keywords(title: str, keywords: List[str]) -> bool:
    """제목에 키워드가 포함되어 있는지 확인"""
    if not title:
        return False
    
    normalized_title = _normalize_tag(title)
    
    for keyword in keywords:
        if _normalize_tag(keyword) in normalized_title:
            return True
    
    return False


def classify_track_type(
    title: str,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None
) -> str:
    """
    노드의 트랙 타입 분류
    
    분류 우선순위:
    1. 태그에 Dialect 관련 키워드가 있으면 → TRACK_B
    2. 태그에 CS 관련 키워드가 있으면 → TRACK_A
    3. 제목에 Dialect 관련 키워드가 있으면 → TRACK_B
    4. 제목에 CS 관련 키워드가 있으면 → TRACK_A
    5. 기본값 → TRACK_A
    
    Args:
        title: 노드 제목
        tags: 노드 태그 리스트
        description: 노드 설명 (선택, 현재 미사용)
        
    Returns:
        'TRACK_A' (CS) 또는 'TRACK_B' (Dialect)
        
    Example:
        >>> classify_track_type("경상도 사투리", ["방언", "지역문화"])
        'TRACK_B'
        
        >>> classify_track_type("SQL 조인", ["데이터베이스", "쿼리"])
        'TRACK_A'
    """
    from knowledge.models import TrackType
    
    tags = tags or []
    
    # 1. 태그 기반 Dialect 판별 (우선순위 높음)
    if _has_matching_tags(tags, DIALECT_TAGS):
        logger.debug(f"[TrackClassifier] '{title}' → TRACK_B (태그 기반)")
        return TrackType.TRACK_B
    
    # 2. 태그 기반 CS 판별
    if _has_matching_tags(tags, CS_TAGS):
        logger.debug(f"[TrackClassifier] '{title}' → TRACK_A (태그 기반)")
        return TrackType.TRACK_A
    
    # 3. 제목 기반 Dialect 판별
    dialect_title_keywords = ['사투리', '방언', '경상도', '전라도', '충청도', '제주도']
    if _title_contains_keywords(title, dialect_title_keywords):
        logger.debug(f"[TrackClassifier] '{title}' → TRACK_B (제목 기반)")
        return TrackType.TRACK_B
    
    # 4. 제목 기반 CS 판별
    cs_title_keywords = ['데이터베이스', 'SQL', '알고리즘', '프로그래밍', '머신러닝', '딥러닝']
    if _title_contains_keywords(title, cs_title_keywords):
        logger.debug(f"[TrackClassifier] '{title}' → TRACK_A (제목 기반)")
        return TrackType.TRACK_A
    
    # 5. 기본값: TRACK_A (기존 도메인)
    logger.debug(f"[TrackClassifier] '{title}' → TRACK_A (기본값)")
    return TrackType.TRACK_A


def classify_batch(nodes: List[dict]) -> List[dict]:
    """
    여러 노드의 트랙 타입을 일괄 분류
    
    Args:
        nodes: [{"title": "...", "tags": [...], ...}, ...]
        
    Returns:
        track_type이 추가된 노드 리스트
    """
    for node in nodes:
        node['track_type'] = classify_track_type(
            title=node.get('title', ''),
            tags=node.get('tags', []),
            description=node.get('description')
        )
    
    return nodes


# =============================================================================
# 통계 함수
# =============================================================================

def get_classification_stats(nodes: List[dict]) -> dict:
    """
    노드 분류 통계
    
    Returns:
        {"TRACK_A": 10, "TRACK_B": 5, "total": 15}
    """
    from knowledge.models import TrackType
    
    stats = {
        TrackType.TRACK_A: 0,
        TrackType.TRACK_B: 0,
    }
    
    for node in nodes:
        track = classify_track_type(
            title=node.get('title', ''),
            tags=node.get('tags', [])
        )
        stats[track] += 1
    
    return {
        "TRACK_A": stats[TrackType.TRACK_A],
        "TRACK_B": stats[TrackType.TRACK_B],
        "total": len(nodes)
    }
