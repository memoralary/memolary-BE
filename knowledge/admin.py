from django.contrib import admin
from .models import KnowledgeNode, KnowledgeEdge

@admin.register(KnowledgeNode)
class KnowledgeNodeAdmin(admin.ModelAdmin):
    list_display = ('title', 'track_type', 'difficulty_index', 'stability_index', 'created_at')
    search_fields = ('title', 'description')
    list_filter = ('track_type', 'cluster_id')

@admin.register(KnowledgeEdge)
class KnowledgeEdgeAdmin(admin.ModelAdmin):
    list_display = ('source', 'target', 'relation_type', 'confidence')
    list_filter = ('relation_type',)
