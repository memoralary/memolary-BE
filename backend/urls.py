"""
URL configuration for backend project.
"""

from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
    SpectacularRedocView,
)

from knowledge.views import UniverseView, TaskStatusView

urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),
    
    # API Documentation (Swagger)
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
    
    # API v1
    path("api/v1/knowledge/", include("knowledge.urls")),
    path("api/v1/analytics/", include("analytics.urls")),
    path("api/v1/debug/", include("debug.urls")),  # 디버깅 API
    path("api/v1/universe/", UniverseView.as_view(), name="universe"),
    path("api/v1/tasks/<str:task_id>/", TaskStatusView.as_view(), name="task-status"),
]
