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
)

urlpatterns = [
    # Ingestion
    path('ingest/', IngestionView.as_view(), name='ingest'),
    
    # CRUD
    path('nodes/', NodeListView.as_view(), name='node-list'),
    path('edges/', EdgeListView.as_view(), name='edge-list'),
]
