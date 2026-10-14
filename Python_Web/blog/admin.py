from django.contrib import admin
from .models import Category, Post


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("id", "name")
    search_fields = ("name",)


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ("title", "category", "status", "created_at", "published_at")
    list_filter = ("status", "category", "created_at")
    search_fields = ("title", "content")
    prepopulated_fields = {"slug": ("title",)}  # 输入标题时自动填充 slug
    date_hierarchy = "created_at"
    list_editable = ("status",)
    actions = ["make_published"]

    @admin.action(description="将选中文章标记为已发布")
    def make_published(self, request, queryset):
        updated = queryset.update(status="published")
        self.message_user(request, f"成功发布 {updated} 篇文章")
