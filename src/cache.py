"""
LRU cache with TTL expiration and deduplication.

Provides query-level caching for the RAG pipeline to avoid redundant
LLM calls and embedding lookups. Tracks hit/miss metrics for the
analytics dashboard.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils import setup_logger, hash_content

logger = setup_logger(__name__)


# ───────────────────────── Cache Metrics ─────────────────────────────

@dataclass
class CacheMetrics:
    """Track cache performance for the analytics dashboard.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of entries evicted due to capacity or TTL.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage (0.0 – 100.0)."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{self.hit_rate:.1f}%",
            "total_requests": self.total_requests,
        }


# ──────────────────────── Query Cache ────────────────────────────────

@dataclass
class CacheEntry:
    """Single cache entry with creation timestamp for TTL checks."""

    value: Any
    created_at: float = field(default_factory=time.time)


class QueryCache:
    """TTL-based LRU cache for query results.

    Evicts least-recently-used entries when capacity is reached, and
    expires entries older than `ttl_seconds`. Thread-safe for single-process
    use (e.g., Streamlit).

    Args:
        max_size: Maximum number of cached entries.
        ttl_seconds: Time-to-live for each entry in seconds.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 3600.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.metrics = CacheMetrics()

    def _make_key(self, query: str, **kwargs: Any) -> str:
        """Generate a deterministic cache key from query + parameters."""
        raw = f"{query}|{sorted(kwargs.items())}"
        return hash_content(raw)

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has exceeded its TTL."""
        return (time.time() - entry.created_at) > self.ttl_seconds

    def _evict_expired(self) -> None:
        """Remove all expired entries from the cache."""
        expired_keys = [
            k for k, v in self._cache.items() if self._is_expired(v)
        ]
        for key in expired_keys:
            del self._cache[key]
            self.metrics.evictions += 1

    def get(self, query: str, **kwargs: Any) -> Optional[Any]:
        """Retrieve a cached result.

        Args:
            query: The query string.
            **kwargs: Additional parameters that affect the cache key.

        Returns:
            Cached value if found and not expired, otherwise None.
        """
        key = self._make_key(query, **kwargs)

        if key in self._cache:
            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                self.metrics.evictions += 1
                self.metrics.misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.metrics.hits += 1
            logger.debug(f"Cache HIT for query: {query[:50]}...")
            return entry.value

        self.metrics.misses += 1
        return None

    def put(self, query: str, value: Any, **kwargs: Any) -> None:
        """Store a result in the cache.

        Args:
            query: The query string.
            value: The result to cache.
            **kwargs: Additional parameters that affect the cache key.
        """
        key = self._make_key(query, **kwargs)

        # Remove if already exists (to update position)
        if key in self._cache:
            del self._cache[key]

        # Evict expired entries
        self._evict_expired()

        # Evict LRU if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            self.metrics.evictions += 1

        self._cache[key] = CacheEntry(value=value)

    def clear(self) -> None:
        """Clear all cached entries. Preserves metrics."""
        self._cache.clear()
        logger.info("Cache cleared")

    @property
    def size(self) -> int:
        return len(self._cache)


# ────────────────────── URL Deduplicator ─────────────────────────────


class URLDeduplicator:
    """Track processed URLs to avoid redundant ingestion.

    Uses content hashing to detect both exact URL matches and
    URLs with minor variations (trailing slashes, query params).
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, url: str) -> bool:
        """Check if a URL has already been processed.

        Args:
            url: URL string to check.

        Returns:
            True if already seen, False otherwise.
        """
        normalized = url.lower().rstrip("/").split("?")[0]
        url_hash = hash_content(normalized)
        return url_hash in self._seen

    def mark_processed(self, url: str) -> None:
        """Mark a URL as processed.

        Args:
            url: URL string to mark.
        """
        normalized = url.lower().rstrip("/").split("?")[0]
        self._seen.add(hash_content(normalized))

    def clear(self) -> None:
        self._seen.clear()
