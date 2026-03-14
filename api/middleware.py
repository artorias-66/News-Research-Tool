"""
Rate limiting, request logging, and API key authentication middleware.
"""

import time
from typing import Optional
from collections import defaultdict

from src.utils import setup_logger


logger = setup_logger(__name__)


class TokenBucketRateLimiter:
    """In-memory token-bucket rate limiter.

    Each client (identified by IP) gets a bucket of `max_tokens` tokens
    that refills at `refill_rate` tokens/second. Each request consumes
    one token.

    Args:
        max_tokens: Maximum burst capacity per client.
        refill_rate: Tokens added per second.
    """

    def __init__(self, max_tokens: int = 20, refill_rate: float = 2.0) -> None:
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": max_tokens, "last_refill": time.time()}
        )

    def _refill(self, client_id: str) -> None:
        bucket = self._buckets[client_id]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(
            self.max_tokens,
            bucket["tokens"] + elapsed * self.refill_rate,
        )
        bucket["last_refill"] = now

    def consume(self, client_id: str) -> bool:
        """Try to consume a token for the given client.

        Args:
            client_id: Client identifier (e.g., IP address).

        Returns:
            True if token consumed, False if rate limited.
        """
        self._refill(client_id)
        bucket = self._buckets[client_id]
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False

    def get_retry_after(self, client_id: str) -> float:
        """Calculate seconds until the next token is available.

        Args:
            client_id: Client identifier.

        Returns:
            Seconds to wait before retrying.
        """
        bucket = self._buckets[client_id]
        tokens_needed = 1.0 - bucket["tokens"]
        return max(0.0, tokens_needed / self.refill_rate)


def validate_api_key(api_key: Optional[str], expected_key: Optional[str]) -> bool:
    """Validate an API key against the expected value.

    Args:
        api_key: The key provided in the request.
        expected_key: The expected key from environment variables.

    Returns:
        True if valid, False otherwise.
    """
    if not expected_key:
        return True  # No auth configured
    return api_key == expected_key
