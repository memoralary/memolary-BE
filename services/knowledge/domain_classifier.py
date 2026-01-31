"""
Domain Classifier - 노드 도메인 분류 및 메타데이터 보강

추출된 노드가 CS(컴퓨터 사이언스)인지 Dialect(사투리)인지 판별하고,
도메인별 메타데이터를 추가합니다.

사용법:
    from services.knowledge.domain_classifier import DomainClassifier
    
    classifier = DomainClassifier()
    result = classifier.classify_and_enrich(nodes)
"""

import os
import logging
from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Constants
# =============================================================================

class Domain(str, Enum):
    CS = "computer_science"
    DIALECT = "dialect"
    UNKNOWN = "unknown"


# CS 도메인 키워드
CS_KEYWORDS = {
    # 프로그래밍 언어
    "python", "java", "javascript", "c++", "c#", "golang", "rust", "typescript",
    "swift", "kotlin", "ruby", "php", "scala", "r", "sql",
    # 기술 스택
    "머신러닝", "딥러닝", "machine learning", "deep learning", "ai", "인공지능",
    "알고리즘", "자료구조", "데이터베이스", "api", "rest", "graphql",
    "프레임워크", "라이브러리", "framework", "library",
    "신경망", "neural network", "cnn", "rnn", "transformer", "bert", "gpt",
    "클라우드", "docker", "kubernetes", "aws", "gcp", "azure",
    "프로그래밍", "코딩", "개발", "백엔드", "프론트엔드", "풀스택",
    "git", "devops", "ci/cd", "agile", "scrum",
    "함수", "클래스", "객체지향", "함수형", "재귀", "반복문",
    "변수", "타입", "컴파일", "인터프리터", "런타임",
    "선형대수", "미적분", "통계", "확률", "최적화",
    "gnn", "gnns", "graph neural network", "link prediction",
}

# 사투리 도메인 키워드
DIALECT_KEYWORDS = {
    # 지역 표현
    "사투리", "방언", "dialect", "억양", "어투",
    "경상도", "전라도", "충청도", "강원도", "제주도", "서울",
    "부산", "대구", "광주", "대전", "인천",
    # 언어학 용어
    "표준어", "비표준어", "구어체", "문어체", "속어", "은어",
    "발음", "어미", "조사", "토씨", "말투",
    # 표현 관련
    "지역어", "향토어", "고어", "옛말", "신조어",
}

# CS 기술 스택 분류
TECH_STACKS = {
    "frontend": ["react", "vue", "angular", "javascript", "typescript", "html", "css"],
    "backend": ["django", "flask", "fastapi", "spring", "node", "express", "nestjs"],
    "data_science": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras"],
    "database": ["mysql", "postgresql", "mongodb", "redis", "elasticsearch"],
    "devops": ["docker", "kubernetes", "aws", "gcp", "azure", "ci/cd", "jenkins"],
    "mobile": ["swift", "kotlin", "flutter", "react native"],
}

# CS 난이도 키워드
DIFFICULTY_KEYWORDS = {
    "beginner": ["기초", "입문", "시작", "처음", "basic", "beginner", "초급", "쉬운"],
    "intermediate": ["중급", "심화", "응용", "intermediate", "중간"],
    "advanced": ["고급", "심층", "전문", "advanced", "expert", "고수준"],
}

# 사투리 지역 키워드
REGION_KEYWORDS = {
    "경상도": ["경상", "부산", "대구", "울산", "경북", "경남", "영남"],
    "전라도": ["전라", "광주", "전북", "전남", "호남"],
    "충청도": ["충청", "대전", "충북", "충남"],
    "강원도": ["강원", "춘천", "강릉"],
    "제주도": ["제주", "탐라"],
    "서울/경기": ["서울", "경기", "수도권"],
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CSMetadata:
    """CS 도메인 메타데이터"""
    tech_stack: List[str] = field(default_factory=list)
    difficulty: str = "intermediate"
    frameworks: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "tech_stack": self.tech_stack,
            "difficulty": self.difficulty,
            "frameworks": self.frameworks,
            "languages": self.languages,
        }


@dataclass
class DialectMetadata:
    """사투리 도메인 메타데이터"""
    region: str = "unknown"
    standard_form: str = ""
    usage_context: List[str] = field(default_factory=list)
    formality: str = "informal"  # formal, informal, slang
    
    def to_dict(self) -> Dict:
        return {
            "region": self.region,
            "standard_form": self.standard_form,
            "usage_context": self.usage_context,
            "formality": self.formality,
        }


@dataclass
class ClassifiedNode:
    """분류된 노드"""
    node_id: str
    title: str
    description: str
    domain: Domain
    confidence: float
    metadata: Dict[str, Any]
    domain_features: List[float]  # GNN용 도메인 피처 벡터


@dataclass
class ClassificationResult:
    """분류 결과"""
    nodes: List[ClassifiedNode]
    cs_count: int
    dialect_count: int
    unknown_count: int
    
    @property
    def total(self) -> int:
        return len(self.nodes)


# =============================================================================
# Domain Classifier
# =============================================================================

class DomainClassifier:
    """
    노드 도메인 분류기
    
    CS 또는 Dialect 도메인을 판별하고 메타데이터를 보강합니다.
    GNN 학습용 도메인 피처 벡터도 생성합니다.
    
    Example:
        classifier = DomainClassifier()
        result = classifier.classify_and_enrich(nodes)
        
        for node in result.nodes:
            print(f"{node.title}: {node.domain.value}")
            print(f"  메타데이터: {node.metadata}")
    """
    
    # 도메인 피처 차원
    DOMAIN_FEATURE_DIM = 16
    
    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: LLM 사용 여부 (더 정확한 분류, 느림)
        """
        self.use_llm = use_llm
    
    def classify_and_enrich(
        self,
        nodes: List[Dict[str, str]]
    ) -> ClassificationResult:
        """
        노드 분류 및 메타데이터 보강
        
        Args:
            nodes: [{"id": "...", "title": "...", "description": "..."}]
            
        Returns:
            ClassificationResult
        """
        classified_nodes = []
        cs_count = 0
        dialect_count = 0
        unknown_count = 0
        
        for node in nodes:
            node_id = node.get("id", "")
            title = node.get("title", "")
            description = node.get("description", "")
            
            # 도메인 분류
            domain, confidence = self._classify_domain(title, description)
            
            # 메타데이터 생성
            if domain == Domain.CS:
                metadata = self._generate_cs_metadata(title, description)
                cs_count += 1
            elif domain == Domain.DIALECT:
                metadata = self._generate_dialect_metadata(title, description)
                dialect_count += 1
            else:
                metadata = {}
                unknown_count += 1
            
            # GNN용 도메인 피처 생성
            domain_features = self._generate_domain_features(domain, metadata)
            
            classified_nodes.append(ClassifiedNode(
                node_id=node_id,
                title=title,
                description=description,
                domain=domain,
                confidence=confidence,
                metadata=metadata,
                domain_features=domain_features
            ))
        
        return ClassificationResult(
            nodes=classified_nodes,
            cs_count=cs_count,
            dialect_count=dialect_count,
            unknown_count=unknown_count
        )
    
    def _classify_domain(
        self,
        title: str,
        description: str
    ) -> tuple[Domain, float]:
        """키워드 기반 도메인 분류"""
        text = f"{title} {description}".lower()
        
        cs_score = 0
        dialect_score = 0
        
        # CS 키워드 매칭
        for keyword in CS_KEYWORDS:
            if keyword.lower() in text:
                cs_score += 1
        
        # Dialect 키워드 매칭
        for keyword in DIALECT_KEYWORDS:
            if keyword.lower() in text:
                dialect_score += 1
        
        total = cs_score + dialect_score
        
        if total == 0:
            return Domain.UNKNOWN, 0.5
        
        if cs_score > dialect_score:
            confidence = cs_score / total
            return Domain.CS, min(confidence, 1.0)
        elif dialect_score > cs_score:
            confidence = dialect_score / total
            return Domain.DIALECT, min(confidence, 1.0)
        else:
            return Domain.UNKNOWN, 0.5
    
    def _generate_cs_metadata(
        self,
        title: str,
        description: str
    ) -> Dict[str, Any]:
        """CS 메타데이터 생성"""
        text = f"{title} {description}".lower()
        
        metadata = CSMetadata()
        
        # 기술 스택 분류
        for stack, keywords in TECH_STACKS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    if stack not in metadata.tech_stack:
                        metadata.tech_stack.append(stack)
                    if stack in ["frontend", "backend", "data_science"]:
                        if keyword not in metadata.frameworks:
                            metadata.frameworks.append(keyword)
        
        # 난이도 판별
        for difficulty, keywords in DIFFICULTY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    metadata.difficulty = difficulty
                    break
        
        # 프로그래밍 언어 탐지
        languages = ["python", "java", "javascript", "c++", "c#", "go", "rust", 
                     "swift", "kotlin", "ruby", "php", "typescript"]
        for lang in languages:
            if lang in text:
                metadata.languages.append(lang)
        
        return metadata.to_dict()
    
    def _generate_dialect_metadata(
        self,
        title: str,
        description: str
    ) -> Dict[str, Any]:
        """사투리 메타데이터 생성"""
        text = f"{title} {description}".lower()
        
        metadata = DialectMetadata()
        
        # 지역 판별
        for region, keywords in REGION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    metadata.region = region
                    break
            if metadata.region != "unknown":
                break
        
        # 사용 맥락 추출
        contexts = []
        context_keywords = {
            "일상": ["일상", "생활", "daily"],
            "감정": ["감정", "기분", "느낌", "emotion"],
            "인사": ["인사", "안녕", "greeting"],
            "욕설": ["욕", "비속어", "slang"],
            "존경": ["존대", "경어", "formal"],
        }
        for context, keywords in context_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    contexts.append(context)
                    break
        metadata.usage_context = contexts
        
        # 격식 판별
        if "비속어" in text or "욕" in text:
            metadata.formality = "slang"
        elif "존대" in text or "경어" in text:
            metadata.formality = "formal"
        
        return metadata.to_dict()
    
    def _generate_domain_features(
        self,
        domain: Domain,
        metadata: Dict[str, Any]
    ) -> List[float]:
        """
        GNN용 도메인 피처 벡터 생성 (16차원)
        
        구조:
        - [0-1]: 도메인 원-핫 (CS, Dialect)
        - [2-5]: CS 기술 스택 (4개 카테고리)
        - [6-8]: CS 난이도 (3단계)
        - [9-12]: 사투리 지역 (4대 권역)
        - [13-15]: 사투리 격식 (3단계)
        """
        features = [0.0] * self.DOMAIN_FEATURE_DIM
        
        # 도메인 원-핫
        if domain == Domain.CS:
            features[0] = 1.0
        elif domain == Domain.DIALECT:
            features[1] = 1.0
        
        if domain == Domain.CS:
            # 기술 스택
            stack_mapping = {"frontend": 2, "backend": 3, "data_science": 4, "devops": 5}
            for stack in metadata.get("tech_stack", []):
                if stack in stack_mapping:
                    features[stack_mapping[stack]] = 1.0
            
            # 난이도
            difficulty = metadata.get("difficulty", "intermediate")
            if difficulty == "beginner":
                features[6] = 1.0
            elif difficulty == "intermediate":
                features[7] = 1.0
            elif difficulty == "advanced":
                features[8] = 1.0
        
        elif domain == Domain.DIALECT:
            # 지역
            region_mapping = {
                "경상도": 9, "전라도": 10, "충청도": 11, 
                "강원도": 11, "제주도": 12, "서울/경기": 9
            }
            region = metadata.get("region", "unknown")
            if region in region_mapping:
                features[region_mapping[region]] = 1.0
            
            # 격식
            formality = metadata.get("formality", "informal")
            if formality == "formal":
                features[13] = 1.0
            elif formality == "informal":
                features[14] = 1.0
            elif formality == "slang":
                features[15] = 1.0
        
        return features
    
    def update_nodes_in_db(
        self,
        classified_nodes: List[ClassifiedNode]
    ) -> int:
        """
        분류 결과를 DB에 업데이트
        
        Returns:
            업데이트된 노드 수
        """
        from django.db import connection
        import json
        
        updated = 0
        
        with connection.cursor() as cursor:
            for node in classified_nodes:
                # 기존 tags에 도메인 정보 추가
                metadata_json = json.dumps({
                    "domain": node.domain.value,
                    "domain_confidence": node.confidence,
                    **node.metadata
                }, ensure_ascii=False)
                
                try:
                    cursor.execute("""
                        UPDATE knowledge_knowledgenode
                        SET tags = %s
                        WHERE id = %s
                    """, [metadata_json, node.node_id])
                    updated += cursor.rowcount
                except Exception as e:
                    logger.error(f"노드 업데이트 실패 ({node.title}): {e}")
        
        logger.info(f"DB에 {updated}개 노드 도메인 정보 저장됨")
        return updated
    
    def get_domain_feature_matrix(
        self,
        classified_nodes: List[ClassifiedNode]
    ) -> "np.ndarray":
        """
        GNN 학습용 도메인 피처 행렬 반환
        
        Returns:
            (N, DOMAIN_FEATURE_DIM) 형태의 numpy 배열
        """
        import numpy as np
        return np.array([n.domain_features for n in classified_nodes])


# =============================================================================
# GNN Integration
# =============================================================================

def create_enhanced_node_features(
    embeddings: "np.ndarray",
    domain_features: "np.ndarray"
) -> "np.ndarray":
    """
    임베딩과 도메인 피처를 결합하여 강화된 노드 피처 생성
    
    Args:
        embeddings: (N, D) 임베딩 행렬
        domain_features: (N, 16) 도메인 피처 행렬
        
    Returns:
        (N, D+16) 강화된 피처 행렬
    """
    import numpy as np
    return np.concatenate([embeddings, domain_features], axis=1)


def get_cross_domain_edge_weight(
    source_domain: Domain,
    target_domain: Domain
) -> float:
    """
    도메인 간 엣지 가중치 반환
    
    같은 도메인 내 연결은 강하게, 다른 도메인 간 연결은 약하게
    """
    if source_domain == target_domain:
        return 1.0
    elif source_domain == Domain.UNKNOWN or target_domain == Domain.UNKNOWN:
        return 0.7
    else:
        # CS <-> Dialect 연결은 약함
        return 0.3


# =============================================================================
# Convenience Functions
# =============================================================================

def classify_nodes(
    nodes: List[Dict[str, str]]
) -> ClassificationResult:
    """
    간편한 노드 분류 함수
    """
    classifier = DomainClassifier()
    return classifier.classify_and_enrich(nodes)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏷️ Domain Classifier 테스트")
    print("=" * 60)
    
    # 테스트 노드
    test_nodes = [
        {"id": "1", "title": "머신러닝", "description": "데이터로부터 패턴을 학습하는 알고리즘"},
        {"id": "2", "title": "딥러닝", "description": "심층 신경망을 활용한 고급 머신러닝 기법"},
        {"id": "3", "title": "React", "description": "프론트엔드 JavaScript 프레임워크"},
        {"id": "4", "title": "경상도 사투리", "description": "부산, 대구 지역에서 사용하는 방언"},
        {"id": "5", "title": "제주도 방언", "description": "제주 지역의 독특한 어휘와 발음"},
        {"id": "6", "title": "안녕하세요", "description": "일반적인 인사말"},
    ]
    
    classifier = DomainClassifier()
    result = classifier.classify_and_enrich(test_nodes)
    
    print(f"\n📊 분류 결과:")
    print(f"   CS: {result.cs_count}개")
    print(f"   Dialect: {result.dialect_count}개")
    print(f"   Unknown: {result.unknown_count}개")
    
    print(f"\n📋 상세:")
    for node in result.nodes:
        domain_icon = {"computer_science": "💻", "dialect": "🗣️", "unknown": "❓"}
        icon = domain_icon.get(node.domain.value, "")
        print(f"\n   {icon} {node.title}")
        print(f"      도메인: {node.domain.value} (신뢰도: {node.confidence:.1%})")
        print(f"      메타데이터: {node.metadata}")
        print(f"      피처 차원: {len(node.domain_features)}")
    
    print("\n🎉 테스트 완료!")
