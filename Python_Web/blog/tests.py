from django.test import TestCase
from django.urls import reverse
from .models import Post, Category


class PostModelTest(TestCase):
    def setUp(self):
        self.category = Category.objects.create(name="技术")
        self.post = Post.objects.create(
            title="第一篇",
            slug="first-post",
            content="内容",
            category=self.category,
            status="published",
        )

    def test_post_str(self):
        self.assertEqual(str(self.post), "第一篇")

    def test_publish(self):
        post = Post.objects.create(title="草稿", slug="draft", content="x")
        self.assertEqual(post.status, "draft")
        post.publish()
        self.assertEqual(post.status, "published")
        self.assertIsNotNone(post.published_at)


class PostViewTest(TestCase):
    def test_list_view(self):
        response = self.client.get(reverse("blog:post_list"))
        self.assertEqual(response.status_code, 200)

    def test_detail_404_for_draft(self):
        Post.objects.create(title="草稿", slug="draft", content="x", status="draft")
        response = self.client.get(
            reverse("blog:post_detail", kwargs={"slug": "draft"})
        )
        self.assertEqual(response.status_code, 404)
