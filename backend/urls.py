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
from django.views.generic import TemplateView

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
    path("api/v1/auth/", include("users.urls")),   # 인증 API
    path("api/v1/universe/", UniverseView.as_view(), name="universe"),
    path("api/v1/tasks/<str:task_id>/", TaskStatusView.as_view(), name="task-status"),

    # Web Push Test Page
    path('push-test/', TemplateView.as_view(template_name='push_test.html')),
    path('sw.js', TemplateView.as_view(template_name='sw.js', content_type='application/javascript')),
]
