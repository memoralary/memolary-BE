from django.contrib import admin
from .models import KnowledgeNode, KnowledgeEdge, KnowledgeCluster, KnowledgeQuiz

@admin.register(KnowledgeNode)
class KnowledgeNodeAdmin(admin.ModelAdmin):
    list_display = ('title', 'track_type', 'difficulty_index', 'stability_index', 'created_at')
    search_fields = ('title', 'description')
    list_filter = ('track_type', 'cluster_id')

@admin.register(KnowledgeEdge)
class KnowledgeEdgeAdmin(admin.ModelAdmin):
    list_display = ('source', 'target', 'relation_type', 'confidence')
    list_filter = ('relation_type',)

@admin.register(KnowledgeCluster)
class KnowledgeClusterAdmin(admin.ModelAdmin):
    list_display = ('name', 'cluster_id', 'node_count', 'is_named', 'created_at')
    list_filter = ('is_named',)
    search_fields = ('name', 'keywords')

@admin.register(KnowledgeQuiz)
class KnowledgeQuizAdmin(admin.ModelAdmin):
    list_display = ('node', 'question', 'answer_index', 'created_at')
    search_fields = ('question', 'node__title')
