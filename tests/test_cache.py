"""
Tests for src/cache.py — LRU cache, TTL expiration, and URL deduplication.
"""

import time
from src.cache import QueryCache, URLDeduplicator


class TestQueryCache:
    """Tests for the TTL-based LRU cache."""

    def test_put_and_get(self, query_cache):
        query_cache.put("test query", {"answer": "42"})
        assert query_cache.get("test query") == {"answer": "42"}

    def test_cache_miss(self, query_cache):
        assert query_cache.get("nonexistent") is None

    def test_lru_eviction(self):
        cache = QueryCache(max_size=2, ttl_seconds=60)
        cache.put("q1", "a1")
        cache.put("q2", "a2")
        cache.put("q3", "a3")  # Should evict q1
        assert cache.get("q1") is None
        assert cache.get("q2") == "a2"
        assert cache.get("q3") == "a3"

    def test_ttl_expiration(self):
        cache = QueryCache(max_size=10, ttl_seconds=0.1)
        cache.put("query", "answer")
        time.sleep(0.2)  # Wait for TTL to expire
        assert cache.get("query") is None

    def test_metrics_tracking(self):
        cache = QueryCache(max_size=10, ttl_seconds=60)
        cache.put("q1", "a1")
        cache.get("q1")  # hit
        cache.get("q2")  # miss

        assert cache.metrics.hits == 1
        assert cache.metrics.misses == 1
        assert cache.metrics.hit_rate == 50.0

    def test_clear(self, query_cache):
        query_cache.put("q1", "a1")
        query_cache.clear()
        assert query_cache.size == 0
        assert query_cache.get("q1") is None

    def test_kwargs_differentiate_keys(self, query_cache):
        query_cache.put("query", "result_openai", provider="openai")
        query_cache.put("query", "result_google", provider="google")
        assert query_cache.get("query", provider="openai") == "result_openai"
        assert query_cache.get("query", provider="google") == "result_google"


class TestURLDeduplicator:
    """Tests for URL deduplication."""

    def test_not_duplicate_initially(self):
        dedup = URLDeduplicator()
        assert dedup.is_duplicate("https://example.com") is False

    def test_duplicate_after_marking(self):
        dedup = URLDeduplicator()
        dedup.mark_processed("https://example.com")
        assert dedup.is_duplicate("https://example.com") is True

    def test_trailing_slash_normalization(self):
        dedup = URLDeduplicator()
        dedup.mark_processed("https://example.com/article/")
        assert dedup.is_duplicate("https://example.com/article") is True

    def test_case_insensitive(self):
        dedup = URLDeduplicator()
        dedup.mark_processed("https://Example.COM/Article")
        assert dedup.is_duplicate("https://example.com/article") is True

    def test_clear(self):
        dedup = URLDeduplicator()
        dedup.mark_processed("https://example.com")
        dedup.clear()
        assert dedup.is_duplicate("https://example.com") is False
