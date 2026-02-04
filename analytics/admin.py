from django.contrib import admin
from .models import User, TestSession, TestResult, SpeechAnalysis, UserDomainStat
from .schedule_models import ReviewSchedule, NotificationLog

class UserDomainStatInline(admin.TabularInline):
    model = UserDomainStat
    extra = 0
    fields = ('domain', 'alpha', 'forgetting_k', 'illusion', 'updated_at')
    readonly_fields = ('updated_at',)

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('username', 'alpha_user', 'base_forgetting_k', 'created_at')
    search_fields = ('username',)
    inlines = [UserDomainStatInline]

@admin.register(ReviewSchedule)
class ReviewScheduleAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'domain', 'scheduled_at', 'status', 'is_manual', 'updated_at')
    list_filter = ('domain', 'status', 'is_manual')
    search_fields = ('user_id', 'note')

@admin.register(TestResult)
class TestResultAdmin(admin.ModelAdmin):
    list_display = ('node', 'is_correct', 'confidence_score', 'response_time_ms', 'test_type')
    list_filter = ('is_correct', 'test_type')

admin.site.register(TestSession)
admin.site.register(SpeechAnalysis)
admin.site.register(NotificationLog)
