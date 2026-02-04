"""
Knowledge API Serializers
"""

from rest_framework import serializers
from knowledge.models import KnowledgeNode, KnowledgeEdge, KnowledgeQuiz


class KnowledgeNodeSerializer(serializers.ModelSerializer):
    """노드 시리얼라이저"""
    
    class Meta:
        model = KnowledgeNode
        fields = [
            'id', 'title', 'description', 'cluster_id',
            'x', 'y', 'z', 'tags', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class KnowledgeQuizSerializer(serializers.ModelSerializer):
    """퀴즈 시리얼라이저"""
    class Meta:
        model = KnowledgeQuiz
        fields = ['id', 'question', 'options', 'answer_index', 'explanation', 'created_at']


class KnowledgeNodeDetailSerializer(serializers.ModelSerializer):
    """노드 상세 조회용 시리얼라이저 (퀴즈 포함)"""
    quiz = serializers.SerializerMethodField()
    
    class Meta:
        model = KnowledgeNode
        fields = [
            'id', 'title', 'description', 'cluster_id',
            'x', 'y', 'z', 'tags', 'created_at',
            'quiz'
        ]
        read_only_fields = ['id', 'created_at']
        
    def get_quiz(self, obj):
        # 가장 최신 퀴즈 하나만 반환
        quiz = obj.quizzes.first()
        if quiz:
            return KnowledgeQuizSerializer(quiz).data
        return None

class KnowledgeEdgeSerializer(serializers.ModelSerializer):
    """엣지 시리얼라이저"""
    source_title = serializers.CharField(source='source.title', read_only=True)
    target_title = serializers.CharField(source='target.title', read_only=True)
    
    class Meta:
        model = KnowledgeEdge
        fields = [
            'id', 'source', 'target', 'source_title', 'target_title',
            'relation_type', 'confidence', 'is_prerequisite', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class IngestionRequestSerializer(serializers.Serializer):
    """Ingestion 요청 시리얼라이저"""
    text = serializers.CharField(required=False, allow_blank=True)
    file = serializers.FileField(required=False)
    async_mode = serializers.BooleanField(default=True)
    
    def validate(self, data):
        if not data.get('text') and not data.get('file'):
            raise serializers.ValidationError(
                "텍스트 또는 파일 중 하나는 필수입니다."
            )
        return data


class TaskStatusSerializer(serializers.Serializer):
    """태스크 상태 시리얼라이저"""
    task_id = serializers.CharField()
    status = serializers.CharField()
    progress = serializers.DictField(required=False)
    result = serializers.DictField(required=False)


class UniverseNodeSerializer(serializers.Serializer):
    """은하수 시각화용 노드 시리얼라이저"""
    id = serializers.UUIDField()
    title = serializers.CharField()
    description = serializers.CharField()
    cluster_id = serializers.CharField(allow_null=True)
    position = serializers.SerializerMethodField()
    tags = serializers.JSONField()
    
    def get_position(self, obj):
        return {
            'x': obj.x or 0,
            'y': obj.y or 0,
            'z': obj.z or 0
        }


class UniverseEdgeSerializer(serializers.Serializer):
    """은하수 시각화용 엣지 시리얼라이저"""
    source = serializers.UUIDField()
    target = serializers.UUIDField()
    relation_type = serializers.CharField()
    confidence = serializers.FloatField()


class UniverseSerializer(serializers.Serializer):
    """은하수 전체 시리얼라이저"""
    nodes = UniverseNodeSerializer(many=True)
    edges = UniverseEdgeSerializer(many=True)
    metadata = serializers.DictField()
