from django.shortcuts import render, get_object_or_404
from .models import Post, Category


def post_list(request):
    """文章列表页"""
    posts = Post.objects.filter(status="published").select_related("category")
    categories = Category.objects.all()
    context = {
        "posts": posts,
        "categories": categories,
    }
    return render(request, "blog/post_list.html", context)


def post_detail(request, slug):
    """文章详情页"""
    post = get_object_or_404(Post, slug=slug, status="published")
    return render(request, "blog/post_detail.html", {"post": post})
