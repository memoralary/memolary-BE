"""
Knowledge Node Extractor

텍스트에서 핵심 지식 개념(Atomic Nodes)을 추출하고,
기존 노드와 비교하여 중복을 제거합니다.

사용법:
    from services.knowledge.extractor import extract_nodes
    
    text = "Machine learning is a subset of AI..."
    existing_titles = ["Artificial Intelligence", "Neural Networks"]
    
    new_nodes = extract_nodes(text, existing_titles)
"""

import logging
import re
from typing import List, Set, Optional
from dataclasses import dataclass, field

from services.llm import get_llm_client, safe_json_parse
from services.llm.schemas import NodeSchema, NodeListResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """너는 지식 공학자(Knowledge Engineer)야. 
주어진 텍스트에서 핵심 지식 개념(Atomic Knowledge Nodes)을 추출해야 해.

## 규칙
1. 각 노드는 **독립적인 학습 단위**여야 함 (하나의 개념 = 하나의 노드)
2. 제목(title)은 **명사형**으로 간결하게 (예: "머신러닝", "역전파 알고리즘")
3. 설명(description)은 해당 개념을 이해하기 위한 핵심 정보만 포함
4. 태그(tags)는 카테고리, 도메인, 관련 키워드 포함
5. 너무 일반적인 개념(예: "컴퓨터", "수학")은 제외
6. 너무 세부적인 개념(예: 특정 변수명)도 제외

## 출력 포맷
반드시 아래 JSON 형식으로만 응답해. 다른 텍스트는 포함하지 마.
{"nodes": [{"title": "개념명", "description": "설명", "tags": ["태그1", "태그2"]}]}
"""

USER_PROMPT_TEMPLATE = """다음 텍스트에서 핵심 지식 개념을 추출해줘:

---
{text}
---

위 텍스트에서 학습에 필요한 핵심 개념들을 JSON 형식으로 추출해줘."""


# =============================================================================
# Deduplication Utilities
# =============================================================================

def normalize_title(title: str) -> str:
    """
    제목을 정규화하여 비교 가능한 형태로 변환
    
    - 소문자 변환
    - 공백/특수문자 제거
    - 일반적인 변형 통일 (예: ML = Machine Learning)
    """
    if not title:
        return ""
    
    normalized = title.lower().strip()
    
    # 공백, 하이픈, 언더스코어 통일
    normalized = re.sub(r'[\s\-_]+', '', normalized)
    
    # 일반적인 약어 확장 (선택적)
    abbreviations = {
        'ml': 'machinelearning',
        'dl': 'deeplearning',
        'ai': 'artificialintelligence',
        'nn': 'neuralnetwork',
        'cnn': 'convolutionalneuralnetwork',
        'rnn': 'recurrentneuralnetwork',
        'nlp': 'naturallanguageprocessing',
        'cv': 'computervision',
    }
    
    if normalized in abbreviations:
        normalized = abbreviations[normalized]
    
    return normalized


def calculate_similarity(title1: str, title2: str) -> float:
    """
    두 제목 간의 유사도 계산 (0.0 ~ 1.0)
    
    간단한 Jaccard 유사도 사용
    """
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    
    if norm1 == norm2:
        return 1.0
    
    # 한쪽이 다른 쪽을 포함하는 경우
    if norm1 in norm2 or norm2 in norm1:
        return 0.8
    
    # 문자 기반 Jaccard 유사도
    set1 = set(norm1)
    set2 = set(norm2)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union


def find_duplicates(
    new_nodes: List[NodeSchema],
    existing_titles: List[str],
    similarity_threshold: float = 0.8
) -> tuple[List[NodeSchema], List[tuple[NodeSchema, str]]]:
    """
    새 노드와 기존 노드 제목을 비교하여 중복 검사
    
    Args:
        new_nodes: 추출된 새 노드 리스트
        existing_titles: 기존 노드 제목 리스트
        similarity_threshold: 중복으로 판단할 유사도 임계값 (0.0 ~ 1.0)
        
    Returns:
        (고유 노드 리스트, 중복 노드와 매칭된 기존 제목 리스트)
    """
    unique_nodes: List[NodeSchema] = []
    duplicates: List[tuple[NodeSchema, str]] = []
    
    # 기존 제목 정규화
    existing_normalized = {
        normalize_title(title): title 
        for title in existing_titles
    }
    
    for node in new_nodes:
        is_duplicate = False
        matched_title = ""
        
        # 정확한 매칭 확인
        node_normalized = normalize_title(node.title)
        if node_normalized in existing_normalized:
            is_duplicate = True
            matched_title = existing_normalized[node_normalized]
        else:
            # 유사도 기반 매칭
            for existing_norm, existing_original in existing_normalized.items():
                similarity = calculate_similarity(node.title, existing_original)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    matched_title = existing_original
                    break
        
        if is_duplicate:
            duplicates.append((node, matched_title))
            logger.debug(f"중복 감지: '{node.title}' ≈ '{matched_title}'")
        else:
            unique_nodes.append(node)
    
    return unique_nodes, duplicates


# =============================================================================
# Main Extraction Function
# =============================================================================

@dataclass
class ExtractionResult:
    """노드 추출 결과"""
    nodes: List[NodeSchema] = field(default_factory=list)
    duplicates: List[tuple[NodeSchema, str]] = field(default_factory=list)
    raw_response: str = ""
    
    @property
    def unique_count(self) -> int:
        return len(self.nodes)
    
    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates)
    
    @property
    def total_extracted(self) -> int:
        return self.unique_count + self.duplicate_count


def extract_nodes(
    text: str,
    existing_titles: Optional[List[str]] = None,
    *,
    similarity_threshold: float = 0.8,
    max_retries: int = 2,
) -> ExtractionResult:
    """
    텍스트에서 지식 노드를 추출하고 중복을 제거합니다.
    
    Args:
        text: 분석할 텍스트
        existing_titles: 기존 노드 제목 리스트 (중복 제거용)
        similarity_threshold: 중복 판단 유사도 임계값 (기본: 0.8)
        max_retries: LLM 호출 실패 시 재시도 횟수
        
    Returns:
        ExtractionResult: 추출된 노드, 중복 노드, 원본 응답 포함
        
    Example:
        result = extract_nodes(
            "Machine learning uses neural networks...",
            existing_titles=["Neural Networks"]
        )
        
        print(f"새 노드: {result.unique_count}")
        print(f"중복 노드: {result.duplicate_count}")
        
        for node in result.nodes:
            print(f"  - {node.title}")
    """
    if not text or not text.strip():
        logger.warning("빈 텍스트가 전달됨")
        return ExtractionResult()
    
    existing_titles = existing_titles or []
    
    # LLM 클라이언트 생성
    client = get_llm_client()
    
    # 프롬프트 구성
    user_prompt = USER_PROMPT_TEMPLATE.format(text=text.strip())
    
    # LLM 호출 (재시도 로직 포함)
    raw_response = ""
    for attempt in range(max_retries + 1):
        try:
            raw_response = client.generate_with_system(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            break
        except Exception as e:
            logger.warning(f"LLM 호출 실패 (시도 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt == max_retries:
                logger.error("최대 재시도 횟수 초과")
                return ExtractionResult(raw_response=str(e))
    
    # JSON 파싱 및 검증
    parsed = safe_json_parse(
        raw_response,
        NodeListResponse,
        return_empty_on_error=True
    )
    
    if not parsed or not parsed.nodes:
        logger.warning("노드 추출 실패 또는 빈 결과")
        return ExtractionResult(raw_response=raw_response)
    
    # 중복 제거
    unique_nodes, duplicates = find_duplicates(
        parsed.nodes,
        existing_titles,
        similarity_threshold
    )
    
    logger.info(
        f"노드 추출 완료: 총 {len(parsed.nodes)}개 → "
        f"고유 {len(unique_nodes)}개, 중복 {len(duplicates)}개"
    )
    
    return ExtractionResult(
        nodes=unique_nodes,
        duplicates=duplicates,
        raw_response=raw_response
    )


def extract_nodes_batch(
    texts: List[str],
    existing_titles: Optional[List[str]] = None,
    *,
    similarity_threshold: float = 0.8,
) -> ExtractionResult:
    """
    여러 텍스트에서 노드를 추출하고 전체 중복을 제거합니다.
    
    Args:
        texts: 분석할 텍스트 리스트
        existing_titles: 기존 노드 제목 리스트
        similarity_threshold: 중복 판단 유사도 임계값
        
    Returns:
        ExtractionResult: 모든 텍스트에서 추출된 고유 노드
    """
    all_nodes: List[NodeSchema] = []
    all_duplicates: List[tuple[NodeSchema, str]] = []
    all_responses: List[str] = []
    
    # 기존 제목 + 추출 과정에서 발견된 제목
    known_titles = list(existing_titles or [])
    
    for i, text in enumerate(texts):
        logger.info(f"텍스트 {i + 1}/{len(texts)} 처리 중...")
        
        result = extract_nodes(
            text,
            existing_titles=known_titles,
            similarity_threshold=similarity_threshold
        )
        
        # 고유 노드의 제목을 known_titles에 추가
        for node in result.nodes:
            known_titles.append(node.title)
        
        all_nodes.extend(result.nodes)
        all_duplicates.extend(result.duplicates)
        all_responses.append(result.raw_response)
    
    return ExtractionResult(
        nodes=all_nodes,
        duplicates=all_duplicates,
        raw_response="\n---\n".join(all_responses)
    )
