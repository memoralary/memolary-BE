"""
Knowledge Models - 지식 그래프 모델

DBML 설계 기반 Django 모델:
- KnowledgeNode: 지식 지도 노드 (3D 시각화 + GNN 학습용)
- KnowledgeEdge: 지식 의존 관계 (GNN 엣지)
"""

import uuid
from django.db import models
from django.core.exceptions import ValidationError


# =============================================================================
# Enums (TextChoices)
# =============================================================================

class TrackType(models.TextChoices):
    """학습 트랙 유형"""
    TRACK_A = 'TRACK_A', 'Track A: 기존 도메인'
    TRACK_B = 'TRACK_B', 'Track B: 신규 도메인'


class RelationType(models.TextChoices):
    """엣지 관계 유형"""
    PREREQUISITE = 'prerequisite', '선행 조건'
    RELATED = 'related', '관련됨'
    PART_OF = 'part_of', '일부'
    INCLUDES = 'includes', '포함'
    EXTENDS = 'extends', '확장'
    SIMILAR_TO = 'similar_to', '유사함'
    CONTRAST = 'contrast', '대조'
    CAUSES = 'causes', '야기함'
    IMPLEMENTS = 'implements', '구현'
    DERIVED_FROM = 'derived_from', '파생'


# =============================================================================
# [2] 지식 노드 모델 (3D Knowledge Graph Nodes)
# =============================================================================

class KnowledgeNode(models.Model):
    """
    지식 그래프의 노드를 나타내는 모델
    
    GNN 학습을 위한 임베딩 벡터와 클러스터링 정보를 포함하며,
    3D 시각화 좌표와 인지 공학 지표를 가집니다.
    
    인지 공학 지표:
    - stability_index (S): 평균 기억 안정성 (유지 기간)
    - difficulty_index (D): 평균 인지 난이도
    """
    id = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False,
        verbose_name='노드 ID',
        help_text='노드의 고유 식별자 (UUID)'
    )
    title = models.CharField(
        max_length=255, 
        unique=True,
        verbose_name='제목',
        help_text='노드의 제목 (유일해야 함)'
    )
    description = models.TextField(
        blank=True, 
        default='',
        verbose_name='설명',
        help_text='노드에 대한 상세 설명'
    )
    embedding = models.BinaryField(
        null=True, 
        blank=True,
        verbose_name='임베딩 벡터',
        help_text='벡터 임베딩 (numpy array를 pickle/bytes로 저장)'
    )
    cluster_id = models.CharField(
        max_length=100, 
        null=True, 
        blank=True,
        db_index=True,
        verbose_name='클러스터 ID',
        help_text='클러스터링 알고리즘에 의해 할당된 클러스터 ID'
    )
    track_type = models.CharField(
        max_length=10,
        choices=TrackType.choices,
        default=TrackType.TRACK_A,
        verbose_name='트랙 유형',
        help_text='A: 기존 도메인, B: 신규 도메인'
    )
    
    # 3D 시각화 좌표 (UMAP 등으로 생성)
    # DBML: pos_x, pos_y, pos_z → 기존 스키마 호환을 위해 x, y, z 사용
    x = models.FloatField(
        null=True,
        blank=True,
        verbose_name='X 좌표',
        help_text='3D 시각화 X 좌표'
    )
    y = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Y 좌표',
        help_text='3D 시각화 Y 좌표'
    )
    z = models.FloatField(
        null=True,
        blank=True,
        verbose_name='Z 좌표',
        help_text='3D 시각화 Z 좌표'
    )
    
    # 인지 공학 지표
    stability_index = models.FloatField(
        default=0.5,
        verbose_name='기억 안정성',
        help_text='S: 평균 기억 안정성 (유지 기간, 0.0~1.0, 기본값 0.5)'
    )
    difficulty_index = models.FloatField(
        default=0.5,
        verbose_name='인지 난이도',
        help_text='D: 평균 인지 난이도 (0.0~1.0, 높을수록 어려움, 기본값 0.5)'
    )
    
    # 메타데이터
    tags = models.JSONField(
        default=list,
        blank=True,
        verbose_name='태그',
        help_text="노드 태그 리스트 예: ['ml', 'optimization']"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='생성 시각',
        help_text='노드 생성 시각'
    )

    class Meta:
        verbose_name = '지식 노드'
        verbose_name_plural = '지식 노드 목록'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['cluster_id']),
            models.Index(fields=['track_type']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return self.title
    
    # =========================================================================
    # 임베딩 관련 메서드
    # =========================================================================
    
    def set_embedding(self, vector):
        """
        numpy array를 BinaryField에 저장
        
        Usage:
            node.set_embedding(np.array([0.1, 0.2, ...]))
        """
        import pickle
        self.embedding = pickle.dumps(vector)
    
    def get_embedding(self):
        """
        BinaryField에서 numpy array로 복원
        
        Usage:
            vector = node.get_embedding()
        """
        import pickle
        if self.embedding:
            return pickle.loads(self.embedding)
        return None
    
    # =========================================================================
    # 인지 공학 메서드
    # =========================================================================
    
    def update_stability_index(self, response_time_ms: int, is_correct: bool) -> float:
        """
        기억 안정성 지수 업데이트
        
        반응 시간(RT)과 정답 여부를 기반으로 노드의 기억 안정성을 계산합니다.
        
        알고리즘 (기본 틀):
        - 빠른 정답: 안정성 증가
        - 느린 정답: 안정성 소폭 증가
        - 오답: 안정성 감소
        
        Args:
            response_time_ms: 인출 반응 시간 (밀리초)
            is_correct: 정답 여부
            
        Returns:
            업데이트된 stability_index 값
            
        Note:
            이 메서드는 기초 틀이며, 실제 인지 공학 모델에 따라 
            수식을 조정해야 합니다.
        """
        # 반응 시간 정규화 (0~1, 10초를 최대로 가정)
        max_rt = 10000  # 10초
        normalized_rt = min(response_time_ms / max_rt, 1.0)
        
        # 기본 가중치
        learning_rate = 0.1
        
        if is_correct:
            # 정답: 빠를수록 안정성 증가
            # stability += learning_rate * (1 - normalized_rt)
            delta = learning_rate * (1 - normalized_rt)
            self.stability_index = min(1.0, self.stability_index + delta)
        else:
            # 오답: 안정성 감소
            delta = learning_rate * 0.5
            self.stability_index = max(0.0, self.stability_index - delta)
        
        self.save(update_fields=['stability_index'])
        return self.stability_index
    
    def update_difficulty_index(self, response_time_ms: int, is_correct: bool) -> float:
        """
        인지 난이도 지수 업데이트
        
        전체 사용자의 반응 시간과 정답률을 기반으로 노드의 난이도를 추정합니다.
        
        Args:
            response_time_ms: 인출 반응 시간 (밀리초)
            is_correct: 정답 여부
            
        Returns:
            업데이트된 difficulty_index 값
            
        Note:
            이 메서드는 기초 틀이며, 실제로는 여러 사용자의 데이터를 
            집계하여 계산해야 합니다.
        """
        # TODO: 전체 사용자 데이터 기반 난이도 계산 구현
        # 현재는 단순 이동 평균 방식
        learning_rate = 0.05
        
        # 정답률 반영
        accuracy_factor = 0.0 if is_correct else 1.0
        
        # RT 반영 (느릴수록 어려움)
        max_rt = 10000
        rt_factor = min(response_time_ms / max_rt, 1.0)
        
        # 가중 평균
        new_difficulty = (accuracy_factor + rt_factor) / 2
        self.difficulty_index = (1 - learning_rate) * self.difficulty_index + learning_rate * new_difficulty
        
        self.save(update_fields=['difficulty_index'])
        return self.difficulty_index
    
    @property
    def cognitive_load_estimate(self) -> str:
        """인지 부하 추정 문자열 반환"""
        if self.difficulty_index >= 0.7:
            return '높음 (High)'
        elif self.difficulty_index >= 0.4:
            return '보통 (Medium)'
        else:
            return '낮음 (Low)'
    
    # DBML 호환을 위한 별칭 프로퍼티 (pos_x, pos_y, pos_z)
    @property
    def pos_x(self):
        """DBML 호환 별칭: x 좌표"""
        return self.x
    
    @pos_x.setter
    def pos_x(self, value):
        self.x = value
    
    @property
    def pos_y(self):
        """DBML 호환 별칭: y 좌표"""
        return self.y
    
    @pos_y.setter
    def pos_y(self, value):
        self.y = value
    
    @property
    def pos_z(self):
        """DBML 호환 별칭: z 좌표"""
        return self.z
    
    @pos_z.setter
    def pos_z(self, value):
        self.z = value


# =============================================================================
# [3] 지식 엣지 모델 (Edges for GNN)
# =============================================================================

class KnowledgeEdge(models.Model):
    """
    지식 그래프의 엣지를 나타내는 모델
    
    두 노드 간의 관계와 메타데이터를 정의합니다.
    GNN 학습 시 메시지 전달에 사용됩니다.
    
    제약 조건:
    - source와 target은 동일할 수 없음 (self-loop 금지)
    - (source, target, relation_type)은 유일해야 함
    """
    id = models.BigAutoField(
        primary_key=True,
        verbose_name='엣지 ID',
        help_text='엣지의 고유 식별자'
    )
    source = models.ForeignKey(
        KnowledgeNode, 
        on_delete=models.CASCADE, 
        related_name='out_edges',
        verbose_name='시작 노드',
        help_text='엣지의 시작 노드 (source → target)'
    )
    target = models.ForeignKey(
        KnowledgeNode, 
        on_delete=models.CASCADE, 
        related_name='in_edges',
        verbose_name='끝 노드',
        help_text='엣지의 끝 노드 (source → target)'
    )
    relation_type = models.CharField(
        max_length=50,
        choices=RelationType.choices,
        default=RelationType.RELATED,
        db_index=True,
        verbose_name='관계 유형',
        help_text='prerequisite, related, part_of 등'
    )
    confidence = models.FloatField(
        default=1.0,
        verbose_name='신뢰도',
        help_text='관계의 신뢰도 (0.0 ~ 1.0)'
    )
    is_prerequisite = models.BooleanField(
        default=False,
        db_index=True,
        verbose_name='선행조건 여부',
        help_text='선행조건 여부 (빠른 필터링용)'
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='생성 시각',
        help_text='엣지 생성 시각'
    )

    class Meta:
        verbose_name = '지식 엣지'
        verbose_name_plural = '지식 엣지 목록'
        ordering = ['-created_at']
        # Constraint: (source, target, relation_type)은 unique
        unique_together = [['source', 'target', 'relation_type']]
        constraints = [
            # source와 target이 같을 수 없음 (Django 6.0+ 문법)
            models.CheckConstraint(
                condition=~models.Q(source=models.F('target')),
                name='no_self_loop'
            )
        ]
        indexes = [
            models.Index(fields=['source', 'target']),
            models.Index(fields=['relation_type']),
            models.Index(fields=['is_prerequisite']),
        ]

    def __str__(self):
        return f"{self.source.title} --[{self.relation_type}]--> {self.target.title}"
    
    def clean(self):
        """
        유효성 검사
        
        제약 조건:
        - source와 target 노드는 동일할 수 없음 (self-loop 금지)
        - confidence는 0.0 ~ 1.0 범위
        """
        if self.source_id and self.target_id and self.source_id == self.target_id:
            raise ValidationError({
                'target': 'Source와 Target 노드는 동일할 수 없습니다. (Self-loop 금지)'
            })
        
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValidationError({
                    'confidence': '신뢰도는 0.0에서 1.0 사이여야 합니다.'
                })
    
    def save(self, *args, **kwargs):
        """
        저장 시 validation 수행 및 is_prerequisite 자동 설정
        """
        self.full_clean()
        
        # relation_type이 'prerequisite'면 is_prerequisite를 True로 자동 설정
        if self.relation_type == RelationType.PREREQUISITE:
            self.is_prerequisite = True
        
        super().save(*args, **kwargs)
    
    @property
    def is_bidirectional(self) -> bool:
        """역방향 엣지가 존재하는지 확인"""
        return KnowledgeEdge.objects.filter(
            source=self.target,
            target=self.source,
            relation_type=self.relation_type
        ).exists()
