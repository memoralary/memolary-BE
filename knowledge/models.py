import uuid
from django.db import models
from django.core.exceptions import ValidationError


class KnowledgeNode(models.Model):
    """
    지식 그래프의 노드를 나타내는 모델
    GNN 학습을 위한 임베딩 벡터와 클러스터링 정보를 포함
    """
    id = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False,
        help_text="노드의 고유 식별자 (UUID)"
    )
    title = models.CharField(
        max_length=255, 
        unique=True,
        help_text="노드의 제목 (유일해야 함)"
    )
    description = models.TextField(
        blank=True, 
        default="",
        help_text="노드에 대한 상세 설명"
    )
    embedding = models.BinaryField(
        null=True, 
        blank=True,
        help_text="벡터 임베딩 (numpy array를 pickle/bytes로 저장)"
    )
    cluster_id = models.CharField(
        max_length=100, 
        null=True, 
        blank=True,
        db_index=True,
        help_text="클러스터링 알고리즘에 의해 할당된 클러스터 ID"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="노드 생성 시각"
    )
    
    # 추가 메타데이터 (선택적)
    tags = models.JSONField(
        default=list,
        blank=True,
        help_text="노드 태그 리스트 예: ['ml', 'optimization']"
    )

    class Meta:
        verbose_name = "Knowledge Node"
        verbose_name_plural = "Knowledge Nodes"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['cluster_id']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return self.title
    
    def set_embedding(self, vector):
        """
        numpy array를 BinaryField에 저장
        Usage: node.set_embedding(np.array([0.1, 0.2, ...]))
        """
        import pickle
        self.embedding = pickle.dumps(vector)
    
    def get_embedding(self):
        """
        BinaryField에서 numpy array로 복원
        Usage: vector = node.get_embedding()
        """
        import pickle
        if self.embedding:
            return pickle.loads(self.embedding)
        return None


class KnowledgeEdge(models.Model):
    """
    지식 그래프의 엣지를 나타내는 모델
    두 노드 간의 관계와 메타데이터를 정의
    """
    RELATION_CHOICES = [
        ('prerequisite', 'Prerequisite'),       # A는 B의 선행조건
        ('related', 'Related'),                  # A와 B는 관련됨
        ('includes', 'Includes'),                # A가 B를 포함함
        ('extends', 'Extends'),                  # A가 B를 확장함
        ('part_of', 'Part Of'),                  # A는 B의 일부
        ('similar_to', 'Similar To'),            # A와 B는 유사함
        ('contrast', 'Contrast'),                # A와 B는 대조됨
        ('causes', 'Causes'),                    # A가 B를 야기함
        ('implements', 'Implements'),            # A가 B를 구현함
        ('derived_from', 'Derived From'),        # A가 B에서 파생됨
    ]
    
    id = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False,
        help_text="엣지의 고유 식별자 (UUID)"
    )
    source = models.ForeignKey(
        KnowledgeNode, 
        on_delete=models.CASCADE, 
        related_name="out_edges",
        help_text="엣지의 시작 노드"
    )
    target = models.ForeignKey(
        KnowledgeNode, 
        on_delete=models.CASCADE, 
        related_name="in_edges",
        help_text="엣지의 끝 노드"
    )
    relation_type = models.CharField(
        max_length=50,
        choices=RELATION_CHOICES,
        db_index=True,
        help_text="관계 유형"
    )
    confidence = models.FloatField(
        default=1.0,
        help_text="관계의 신뢰도 (0.0 ~ 1.0)"
    )
    is_prerequisite = models.BooleanField(
        default=False,
        db_index=True,
        help_text="선행조건 여부 (빠른 필터링용)"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="엣지 생성 시각"
    )

    class Meta:
        verbose_name = "Knowledge Edge"
        verbose_name_plural = "Knowledge Edges"
        ordering = ['-created_at']
        # Constraint: (source, target, relation_type)은 unique
        unique_together = [['source', 'target', 'relation_type']]
        indexes = [
            models.Index(fields=['source', 'target']),
            models.Index(fields=['relation_type']),
            models.Index(fields=['is_prerequisite']),
        ]

    def __str__(self):
        return f"{self.source.title} --[{self.relation_type}]--> {self.target.title}"
    
    def clean(self):
        """
        Constraint: source와 target은 동일할 수 없음
        """
        if self.source_id == self.target_id:
            raise ValidationError({
                'target': 'Source와 Target 노드는 동일할 수 없습니다.'
            })
    
    def save(self, *args, **kwargs):
        """
        저장 시 validation 수행 및 is_prerequisite 자동 설정
        """
        self.full_clean()
        # relation_type이 'prerequisite'면 is_prerequisite를 True로 자동 설정
        if self.relation_type == 'prerequisite':
            self.is_prerequisite = True
        super().save(*args, **kwargs)
