"""
Cluster Naming Service - LLM 기반 클러스터 자동 네이밍

클러스터에 속한 노드들의 title, description을 분석하여
OpenAI API를 통해 적절한 클러스터 이름을 생성합니다.

사용법:
    from services.knowledge.cluster_naming import ClusterNamingService
    
    service = ClusterNamingService()
    
    # 단일 클러스터 이름 생성
    name = service.generate_cluster_name(cluster_id="cluster_001")
    
    # 이름이 없는 모든 클러스터 배치 처리
    results = service.batch_generate_names(min_nodes=5)
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Naming Result
# =============================================================================

@dataclass
class NamingResult:
    """클러스터 네이밍 결과"""
    cluster_id: str
    name: str
    keywords: List[str]
    node_count: int
    success: bool
    error: Optional[str] = None


# =============================================================================
# Cluster Naming Service
# =============================================================================

class ClusterNamingService:
    """
    LLM 기반 클러스터 네이밍 서비스
    
    클러스터에 속한 노드들의 제목과 설명을 분석하여
    적절한 클러스터 이름과 키워드를 생성합니다.
    
    Args:
        min_nodes_for_naming: 이름 생성에 필요한 최소 노드 수 (기본: 5)
        model: 사용할 OpenAI 모델 (기본: gpt-4o-mini)
    """
    
    DEFAULT_MIN_NODES = 5
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(
        self,
        min_nodes_for_naming: int = DEFAULT_MIN_NODES,
        model: str = DEFAULT_MODEL
    ):
        self.min_nodes = min_nodes_for_naming
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """OpenAI 클라이언트 (lazy loading)"""
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai 패키지가 설치되지 않았습니다. pip install openai")
        return self._client
    
    def _build_prompt(self, nodes: List[Dict[str, str]]) -> str:
        """
        LLM에 전달할 프롬프트 생성
        
        Args:
            nodes: [{"title": "...", "description": "..."}, ...]
        """
        node_list = "\n".join([
            f"- {n['title']}: {n['description'][:150]}..."
            if len(n.get('description', '')) > 150
            else f"- {n['title']}: {n.get('description', '(설명 없음)')}"
            for n in nodes
        ])
        
        prompt = f"""당신은 지식 분류 전문가입니다.
다음은 하나의 클러스터에 속한 지식 노드들입니다.
이 노드들의 공통점을 분석하여 클러스터의 이름을 지어주세요.

## 노드 목록
{node_list}

## 요청 사항
1. 클러스터 이름: 2-5단어의 간결하고 명확한 이름
2. 핵심 키워드: 3-5개의 관련 키워드

## 응답 형식 (JSON)
{{"name": "클러스터 이름", "keywords": ["키워드1", "키워드2", "키워드3"]}}

응답:"""
        return prompt
    
    def _parse_response(self, response_text: str) -> Tuple[str, List[str]]:
        """
        LLM 응답 파싱
        
        Returns:
            (name, keywords)
        """
        import json
        
        # JSON 파싱 시도
        try:
            # 응답에서 JSON 부분만 추출
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                return data.get("name", ""), data.get("keywords", [])
        except json.JSONDecodeError:
            pass
        
        # JSON 파싱 실패 시 첫 줄을 이름으로 사용
        lines = response_text.strip().split("\n")
        name = lines[0].strip().strip('"')
        return name, []
    
    def generate_cluster_name(self, cluster_id: str) -> NamingResult:
        """
        단일 클러스터의 이름 생성
        
        Args:
            cluster_id: 클러스터 ID
            
        Returns:
            NamingResult
        """
        from knowledge.models import KnowledgeNode, KnowledgeCluster
        
        # 클러스터에 속한 노드들 조회
        nodes = KnowledgeNode.objects.filter(cluster_id=cluster_id).values('title', 'description')
        nodes_list = list(nodes)
        
        if len(nodes_list) < self.min_nodes:
            return NamingResult(
                cluster_id=cluster_id,
                name="",
                keywords=[],
                node_count=len(nodes_list),
                success=False,
                error=f"노드 수 부족 ({len(nodes_list)} < {self.min_nodes})"
            )
        
        # 프롬프트 생성 및 LLM 호출
        prompt = self._build_prompt(nodes_list)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 지식 분류 전문가입니다. JSON 형식으로만 응답하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            response_text = response.choices[0].message.content
            name, keywords = self._parse_response(response_text)
            
            # DB 저장
            cluster, created = KnowledgeCluster.objects.update_or_create(
                cluster_id=cluster_id,
                defaults={
                    "name": name,
                    "keywords": keywords,
                    "node_count": len(nodes_list),
                    "is_named": True
                }
            )
            
            logger.info(f"클러스터 이름 생성 완료: {cluster_id} → {name}")
            
            return NamingResult(
                cluster_id=cluster_id,
                name=name,
                keywords=keywords,
                node_count=len(nodes_list),
                success=True
            )
            
        except Exception as e:
            logger.exception(f"클러스터 네이밍 실패: {cluster_id}")
            return NamingResult(
                cluster_id=cluster_id,
                name="",
                keywords=[],
                node_count=len(nodes_list),
                success=False,
                error=str(e)
            )
    
    def batch_generate_names(
        self,
        min_nodes: Optional[int] = None,
        force_regenerate: bool = False
    ) -> List[NamingResult]:
        """
        이름이 없는 모든 클러스터에 대해 배치로 이름 생성
        
        Args:
            min_nodes: 최소 노드 수 (기본: self.min_nodes)
            force_regenerate: True면 이미 이름이 있는 클러스터도 재생성
            
        Returns:
            NamingResult 리스트
        """
        from knowledge.models import KnowledgeNode, KnowledgeCluster
        from django.db.models import Count
        
        min_nodes = min_nodes or self.min_nodes
        
        # 클러스터별 노드 수 집계
        cluster_stats = (
            KnowledgeNode.objects
            .values('cluster_id')
            .annotate(node_count=Count('id'))
            .filter(cluster_id__isnull=False, node_count__gte=min_nodes)
        )
        
        results = []
        
        for stat in cluster_stats:
            cluster_id = stat['cluster_id']
            
            # 이미 이름이 있는 경우 스킵 (force가 아니면)
            if not force_regenerate:
                try:
                    existing = KnowledgeCluster.objects.get(cluster_id=cluster_id)
                    if existing.is_named:
                        logger.debug(f"이미 이름이 있는 클러스터 스킵: {cluster_id}")
                        continue
                except KnowledgeCluster.DoesNotExist:
                    pass
            
            # 이름 생성
            result = self.generate_cluster_name(cluster_id)
            results.append(result)
        
        # 결과 요약 로깅
        success_count = sum(1 for r in results if r.success)
        logger.info(f"배치 네이밍 완료: {success_count}/{len(results)} 성공")
        
        return results
    
    def get_cluster_summary(self) -> Dict:
        """
        전체 클러스터 현황 요약
        
        Returns:
            {"total": int, "named": int, "unnamed": int, "clusters": [...]}
        """
        from knowledge.models import KnowledgeNode, KnowledgeCluster
        from django.db.models import Count
        
        # 클러스터별 노드 수
        cluster_stats = (
            KnowledgeNode.objects
            .values('cluster_id')
            .annotate(node_count=Count('id'))
            .filter(cluster_id__isnull=False)
            .order_by('-node_count')
        )
        
        clusters = []
        named_count = 0
        
        for stat in cluster_stats:
            cluster_id = stat['cluster_id']
            node_count = stat['node_count']
            
            try:
                cluster = KnowledgeCluster.objects.get(cluster_id=cluster_id)
                name = cluster.name
                is_named = cluster.is_named
            except KnowledgeCluster.DoesNotExist:
                name = ""
                is_named = False
            
            if is_named:
                named_count += 1
            
            clusters.append({
                "cluster_id": cluster_id,
                "name": name,
                "node_count": node_count,
                "is_named": is_named,
                "ready_for_naming": node_count >= self.min_nodes
            })
        
        return {
            "total": len(clusters),
            "named": named_count,
            "unnamed": len(clusters) - named_count,
            "min_nodes_required": self.min_nodes,
            "clusters": clusters
        }
