"""
Knowledge API URLs
"""

from django.urls import path
from knowledge.views import (
    IngestionView,
    TaskStatusView,
    UniverseView,
    NodeListView,
    EdgeListView,
    RecommendView,
    GenerateQuizView,
    NodeDetailView,
)

urlpatterns = [
    # Ingestion
    path('ingest/', IngestionView.as_view(), name='ingest'),
    
    # CRUD
    path('nodes/', NodeListView.as_view(), name='node-list'),
    path('nodes/<uuid:pk>/', NodeDetailView.as_view(), name='node-detail'),
    path('edges/', EdgeListView.as_view(), name='edge-list'),
    
    # Recommend (GNN Integration)
    path('recommend/', RecommendView.as_view(), name='recommend'),
    # Quiz
    path('nodes/<uuid:node_id>/quiz/', GenerateQuizView.as_view(), name='generate-quiz'),
]
