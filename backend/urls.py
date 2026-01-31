"""
URL configuration for backend project.
"""

from django.contrib import admin
from django.urls import path, include

from knowledge.views import UniverseView, TaskStatusView

urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),
    
    # API v1
    path("api/v1/knowledge/", include("knowledge.urls")),
    path("api/v1/universe/", UniverseView.as_view(), name="universe"),
    path("api/v1/tasks/<str:task_id>/", TaskStatusView.as_view(), name="task-status"),
]
