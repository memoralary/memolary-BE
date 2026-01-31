"""
Curriculum Designer - Prerequisite Edge Generator

노드 리스트에서 학습 순서(Prerequisite) 관계를 추출하고,
DAG(Directed Acyclic Graph)를 보장합니다.

사용법:
    from services.knowledge.curriculum import generate_prerequisites
    
    nodes = [
        {"title": "Linear Algebra", "description": "..."},
        {"title": "Machine Learning", "description": "..."},
    ]
    
    edges = generate_prerequisites(nodes)
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from services.llm import get_llm_client, safe_json_parse
from services.llm.schemas import EdgeSchema, EdgeListResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """너는 커리큘럼 설계자(Curriculum Designer)야.
주어진 지식 개념(노드) 리스트를 보고 학습 순서(Prerequisite 관계)를 결정해야 해.

## 판단 기준
**"A를 모르면 B를 이해할 수 없는가?"**가 참이면 A → B 엣지를 생성해.

예시:
- "선형대수" → "머신러닝" (선형대수 없이 머신러닝 이해 불가)
- "미적분" → "역전파" (미적분 없이 역전파 이해 불가)

## 규칙
1. 명확한 선행 관계만 추출해. 애매하면 생성하지 마.
2. 각 관계에 신뢰도(confidence)를 0.0~1.0 사이로 부여해.
   - 1.0: 절대적 선행조건 (A 없이 B 이해 불가능)
   - 0.8~0.9: 강한 선행조건 (A 없이 B 이해 매우 어려움)
   - 0.6~0.7: 약한 선행조건 (A 알면 B 이해에 도움)
   - 0.5 미만: 생성하지 마
3. 사이클을 만들지 마 (A→B→C→A 금지)
4. 자기 자신으로의 엣지 금지 (A→A 금지)

## 출력 포맷
반드시 아래 JSON 형식으로만 응답해. 다른 텍스트는 포함하지 마.
{"edges": [{"source": "소스 노드 제목", "target": "타겟 노드 제목", "relation_type": "prerequisite", "confidence": 0.9}]}

선행 관계가 없으면 빈 리스트를 반환해: {"edges": []}
"""

USER_PROMPT_TEMPLATE = """다음 지식 개념들 사이의 학습 선행조건(Prerequisite) 관계를 분석해줘:

---
{nodes_text}
---

각 개념 쌍에 대해 "A를 모르면 B를 이해할 수 없는가?"를 판단하고,
참이면 A → B 엣지를 JSON 형식으로 출력해줘."""


# =============================================================================
# DAG Validation & Cycle Removal
# =============================================================================

def detect_cycles(edges: List[EdgeSchema]) -> List[List[str]]:
    """
    그래프에서 모든 사이클을 탐지합니다.
    
    DFS 기반 사이클 탐지
    
    Returns:
        발견된 사이클 리스트 (각 사이클은 노드 제목 리스트)
    """
    # 인접 리스트 구성
    graph: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        graph[edge.source].append(edge.target)
    
    # 모든 노드 수집
    all_nodes = set(graph.keys())
    for edge in edges:
        all_nodes.add(edge.target)
    
    cycles = []
    visited = set()
    rec_stack = set()
    path = []
    
    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                # 사이클 발견
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
                return True
        
        path.pop()
        rec_stack.remove(node)
        return False
    
    for node in all_nodes:
        if node not in visited:
            dfs(node)
    
    return cycles


def remove_cycles(edges: List[EdgeSchema]) -> Tuple[List[EdgeSchema], List[EdgeSchema]]:
    """
    사이클이 발견되면 신뢰도가 낮은 엣지를 제거하여 DAG를 보장합니다.
    
    Returns:
        (DAG 엣지 리스트, 제거된 엣지 리스트)
    """
    remaining_edges = list(edges)
    removed_edges = []
    
    max_iterations = len(edges)  # 무한 루프 방지
    
    for _ in range(max_iterations):
        cycles = detect_cycles(remaining_edges)
        
        if not cycles:
            break
        
        # 첫 번째 사이클 처리
        cycle = cycles[0]
        logger.warning(f"사이클 발견: {' → '.join(cycle)}")
        
        # 사이클 내 엣지 찾기
        cycle_edges = []
        for i in range(len(cycle) - 1):
            source, target = cycle[i], cycle[i + 1]
            for edge in remaining_edges:
                if edge.source == source and edge.target == target:
                    cycle_edges.append(edge)
                    break
        
        if not cycle_edges:
            logger.error("사이클 엣지를 찾을 수 없음")
            break
        
        # 신뢰도가 가장 낮은 엣지 제거
        weakest_edge = min(cycle_edges, key=lambda e: e.confidence)
        remaining_edges.remove(weakest_edge)
        removed_edges.append(weakest_edge)
        
        logger.info(
            f"사이클 제거: {weakest_edge.source} → {weakest_edge.target} "
            f"(confidence: {weakest_edge.confidence})"
        )
    
    return remaining_edges, removed_edges


def topological_sort(edges: List[EdgeSchema]) -> List[str]:
    """
    DAG의 위상 정렬을 수행합니다.
    
    Returns:
        학습 순서대로 정렬된 노드 제목 리스트
    """
    # 인접 리스트와 진입 차수 계산
    graph: Dict[str, List[str]] = defaultdict(list)
    in_degree: Dict[str, int] = defaultdict(int)
    
    all_nodes = set()
    for edge in edges:
        graph[edge.source].append(edge.target)
        in_degree[edge.target] += 1
        all_nodes.add(edge.source)
        all_nodes.add(edge.target)
    
    # 진입 차수 0인 노드부터 시작
    queue = [node for node in all_nodes if in_degree[node] == 0]
    result = []
    
    while queue:
        # 알파벳 순으로 정렬하여 일관성 유지
        queue.sort()
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result


# =============================================================================
# Main Functions
# =============================================================================

@dataclass
class PrerequisiteResult:
    """선행조건 추출 결과"""
    edges: List[EdgeSchema] = field(default_factory=list)
    removed_edges: List[EdgeSchema] = field(default_factory=list)
    learning_order: List[str] = field(default_factory=list)
    raw_response: str = ""
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    @property
    def removed_count(self) -> int:
        return len(self.removed_edges)
    
    @property
    def is_dag(self) -> bool:
        """DAG 여부 확인"""
        return len(detect_cycles(self.edges)) == 0
    
    def to_dict_list(self) -> List[Dict]:
        """JSON 직렬화 가능한 딕셔너리 리스트 반환"""
        return [
            {
                "source_title": edge.source,
                "target_title": edge.target,
                "relation": edge.relation_type,
                "confidence": edge.confidence
            }
            for edge in self.edges
        ]


def generate_prerequisites(
    nodes: List[Dict[str, str]],
    *,
    min_confidence: float = 0.5,
    ensure_dag: bool = True,
    max_retries: int = 2,
) -> PrerequisiteResult:
    """
    노드 리스트에서 선행조건(Prerequisite) 관계를 추출합니다.
    
    Args:
        nodes: 노드 딕셔너리 리스트 [{"title": "...", "description": "..."}, ...]
        min_confidence: 최소 신뢰도 (이하는 필터링)
        ensure_dag: True이면 사이클 제거하여 DAG 보장
        max_retries: LLM 호출 실패 시 재시도 횟수
        
    Returns:
        PrerequisiteResult: 엣지, 제거된 엣지, 학습 순서 포함
        
    Example:
        nodes = [
            {"title": "선형대수", "description": "행렬과 벡터 연산"},
            {"title": "머신러닝", "description": "데이터 기반 학습"},
            {"title": "딥러닝", "description": "심층 신경망"},
        ]
        
        result = generate_prerequisites(nodes)
        
        for edge in result.edges:
            print(f"{edge.source} → {edge.target} ({edge.confidence})")
        
        print(f"학습 순서: {result.learning_order}")
    """
    if not nodes or len(nodes) < 2:
        logger.warning("노드가 2개 미만이면 관계를 생성할 수 없습니다.")
        return PrerequisiteResult()
    
    # 노드 텍스트 구성
    nodes_text = "\n".join([
        f"- **{node.get('title', 'Untitled')}**: {node.get('description', '(설명 없음)')[:100]}"
        for node in nodes
    ])
    
    # LLM 클라이언트 생성
    client = get_llm_client()
    
    # 프롬프트 구성
    user_prompt = USER_PROMPT_TEMPLATE.format(nodes_text=nodes_text)
    
    # LLM 호출
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
                return PrerequisiteResult(raw_response=str(e))
    
    # JSON 파싱
    parsed = safe_json_parse(
        raw_response,
        EdgeListResponse,
        return_empty_on_error=True
    )
    
    if not parsed or not parsed.edges:
        logger.info("선행조건 관계가 없거나 파싱 실패")
        return PrerequisiteResult(raw_response=raw_response)
    
    # 유효한 노드 제목 집합
    valid_titles = {node.get('title', '') for node in nodes}
    
    # 필터링: 유효한 노드만, 자기 참조 제외, 최소 신뢰도 이상
    filtered_edges = []
    for edge in parsed.edges:
        if edge.source not in valid_titles:
            logger.debug(f"유효하지 않은 source: {edge.source}")
            continue
        if edge.target not in valid_titles:
            logger.debug(f"유효하지 않은 target: {edge.target}")
            continue
        if edge.source == edge.target:
            logger.debug(f"자기 참조 제외: {edge.source}")
            continue
        if edge.confidence < min_confidence:
            logger.debug(f"신뢰도 부족: {edge.source} → {edge.target} ({edge.confidence})")
            continue
        
        # relation_type을 prerequisite로 강제
        edge.relation_type = "prerequisite"
        filtered_edges.append(edge)
    
    # DAG 보장
    removed_edges = []
    if ensure_dag:
        filtered_edges, removed_edges = remove_cycles(filtered_edges)
    
    # 학습 순서 계산
    learning_order = topological_sort(filtered_edges)
    
    logger.info(
        f"선행조건 추출 완료: {len(filtered_edges)}개 엣지, "
        f"{len(removed_edges)}개 제거, DAG: {len(detect_cycles(filtered_edges)) == 0}"
    )
    
    return PrerequisiteResult(
        edges=filtered_edges,
        removed_edges=removed_edges,
        learning_order=learning_order,
        raw_response=raw_response
    )


def generate_prerequisites_from_titles(
    titles: List[str],
    **kwargs
) -> PrerequisiteResult:
    """
    제목 리스트만으로 선행조건을 추출합니다.
    
    Args:
        titles: 노드 제목 리스트
        **kwargs: generate_prerequisites에 전달할 추가 인자
        
    Returns:
        PrerequisiteResult
    """
    nodes = [{"title": title, "description": ""} for title in titles]
    return generate_prerequisites(nodes, **kwargs)
