"""
LLM 응답 검증을 위한 Pydantic 스키마

이 모듈은 LLM 응답을 구조화된 데이터로 변환할 때 사용하는
Pydantic 모델을 정의합니다.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Knowledge Graph 관련 스키마
# =============================================================================

class NodeSchema(BaseModel):
    """지식 그래프 노드 스키마"""
    id: str = Field(..., description="노드의 고유 식별자")
    title: str = Field(..., min_length=1, max_length=255, description="노드 제목")
    description: Optional[str] = Field(default="", description="노드 설명")
    tags: List[str] = Field(default_factory=list, description="태그 리스트")
    
    @field_validator("title")
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title은 비어있을 수 없습니다.")
        return v.strip()
    
    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


class EdgeSchema(BaseModel):
    """지식 그래프 엣지 스키마"""
    source: str = Field(..., description="소스 노드 ID")
    target: str = Field(..., description="타겟 노드 ID")
    relation_type: str = Field(..., description="관계 유형")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="신뢰도")
    
    @field_validator("relation_type")
    @classmethod
    def relation_type_must_be_valid(cls, v: str) -> str:
        valid_types = {
            "prerequisite", "related", "includes", "extends",
            "part_of", "similar_to", "contrast", "causes",
            "implements", "derived_from"
        }
        v_lower = v.lower().replace(" ", "_")
        if v_lower not in valid_types:
            # 유효하지 않으면 'related'로 기본 설정
            return "related"
        return v_lower


class NodeListResponse(BaseModel):
    """노드 리스트 응답 스키마"""
    nodes: List[NodeSchema] = Field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.nodes)


class EdgeListResponse(BaseModel):
    """엣지 리스트 응답 스키마"""
    edges: List[EdgeSchema] = Field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.edges)


class GraphResponse(BaseModel):
    """전체 그래프 응답 스키마"""
    nodes: List[NodeSchema] = Field(default_factory=list)
    edges: List[EdgeSchema] = Field(default_factory=list)
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)


# =============================================================================
# 일반적인 LLM 응답 스키마
# =============================================================================

class TextListResponse(BaseModel):
    """텍스트 리스트 응답"""
    items: List[str] = Field(default_factory=list)


class KeyValueResponse(BaseModel):
    """키-값 쌍 응답"""
    data: dict = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """분석 결과 응답"""
    summary: str = Field(default="", description="분석 요약")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
