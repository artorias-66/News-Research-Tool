"""
Utility functions for the Equity Research Tool.

Provides text cleaning, URL validation, performance tracking,
and logging configuration.
"""

import re
import time
import logging
import hashlib
import functools
from typing import Callable, Any
from urllib.parse import urlparse

from src.exceptions import URLValidationError


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a named logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


logger = setup_logger(__name__)


def clean_text(text: str) -> str:
    """Remove boilerplate noise and normalize whitespace from article content.

    Strips common web UI patterns (navigation, footers, CTAs) and filters
    out very short lines that are typically non-content elements.

    Args:
        text: Raw article text to clean.

    Returns:
        Cleaned text with noise patterns removed and whitespace normalized.
    """
    if not text:
        return ""

    noise_patterns = [
        r"\bTags\b",
        r"\bShare\b",
        r"\bTweet\b",
        r"\bFollow\b",
        r"\bSubscribe\b",
        r"\bAdvertisement\b",
        r"\bCookie Policy\b",
        r"\bPrivacy Policy\b",
        r"\bRead More\b",
        r"\bClick Here\b",
        r"\bComments\b",
        r"\bRelated Articles\b",
        r"\bSign up\b",
        r"\bLog in\b",
        r"\bNewsletter\b",
        r"\bSponsored\b",
        r"\bShare this article\b",
        r"\bCopy link\b",
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" +", " ", text)

    # Filter out short non-content lines
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line and (len(line) > 30 or line.endswith(".") or line.endswith("?")):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def validate_url(url: str) -> str:
    """Validate and normalize a URL.

    Args:
        url: URL string to validate.

    Returns:
        Normalized URL string.

    Raises:
        URLValidationError: If the URL is malformed or uses an unsupported scheme.
    """
    url = url.strip()

    if not url:
        raise URLValidationError(url, "URL cannot be empty.")

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise URLValidationError(
            url, f"Unsupported scheme '{parsed.scheme}'. Use http or https."
        )

    if not parsed.netloc:
        raise URLValidationError(url, "Missing domain name.")

    # Basic domain format check
    if "." not in parsed.netloc:
        raise URLValidationError(url, f"Invalid domain '{parsed.netloc}'.")

    return url


def hash_content(content: str) -> str:
    """Generate an MD5 hash of content for deduplication.

    Args:
        content: String content to hash.

    Returns:
        Hex-encoded MD5 hash string.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def timed(func: Callable) -> Callable:
    """Decorator that logs execution time of a function.

    Tracks wall-clock time and logs it at INFO level. Also stores
    the elapsed time as a `_last_elapsed` attribute on the function
    for programmatic access (e.g., analytics dashboard).

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that tracks execution time.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        wrapper._last_elapsed = elapsed
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result

    wrapper._last_elapsed = 0.0
    return wrapper


def truncate_text(text: str, max_length: int = 2000, suffix: str = "...") -> str:
    """Truncate text to a maximum length with a suffix.

    Args:
        text: Text to truncate.
        max_length: Maximum character count before truncation.
        suffix: String to append when truncated.

    Returns:
        Original text if within limit, otherwise truncated with suffix.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
