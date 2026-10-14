from django.db import models
from django.utils import timezone


class Category(models.Model):
    """文章分类"""

    name = models.CharField("分类名", max_length=50, unique=True)

    class Meta:
        verbose_name = "分类"
        verbose_name_plural = "分类"

    def __str__(self):
        return self.name


class Post(models.Model):
    """博客文章"""

    STATUS_CHOICES = [
        ("draft", "草稿"),
        ("published", "已发布"),
    ]

    title = models.CharField("标题", max_length=200)
    slug = models.SlugField("URL 标识", max_length=200, unique=True)
    content = models.TextField("正文")
    category = models.ForeignKey(
        Category,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="posts",
        verbose_name="分类",
    )
    status = models.CharField(
        "状态", max_length=10, choices=STATUS_CHOICES, default="draft"
    )
    created_at = models.DateTimeField("创建时间", auto_now_add=True)
    updated_at = models.DateTimeField("更新时间", auto_now=True)
    published_at = models.DateTimeField("发布时间", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "文章"
        verbose_name_plural = "文章"

    def __str__(self):
        return self.title

    def publish(self):
        """发布文章"""
        self.status = "published"
        self.published_at = timezone.now()
        self.save()
