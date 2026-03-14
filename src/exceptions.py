"""
Custom exception hierarchy for the Equity Research Tool.

Provides structured, recoverable error handling with error codes,
user-friendly messages, and recovery suggestions.
"""

from typing import Optional


class ResearchToolError(Exception):
    """Base exception for all Research Tool errors.

    Attributes:
        message: Human-readable error description.
        error_code: Machine-readable error identifier.
        recovery_hint: Suggested user action to resolve the error.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        recovery_hint: Optional[str] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.recovery_hint = recovery_hint or "Please try again or contact support."
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Serialize exception for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "recovery_hint": self.recovery_hint,
        }


class URLFetchError(ResearchToolError):
    """Raised when URL content cannot be fetched."""

    def __init__(
        self,
        url: str,
        status_code: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        default_msg = f"Failed to fetch content from {url}"
        if status_code:
            default_msg += f" (HTTP {status_code})"
        super().__init__(
            message=message or default_msg,
            error_code="URL_FETCH_ERROR",
            recovery_hint="Check the URL is accessible and try again. Ensure the site isn't blocking automated requests.",
        )


class URLValidationError(ResearchToolError):
    """Raised when a URL fails validation."""

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        super().__init__(
            message=f"Invalid URL '{url}': {reason}",
            error_code="URL_VALIDATION_ERROR",
            recovery_hint="Provide a valid HTTP/HTTPS URL (e.g., https://example.com/article).",
        )


class EmbeddingError(ResearchToolError):
    """Raised when embedding generation fails."""

    def __init__(self, message: Optional[str] = None) -> None:
        super().__init__(
            message=message or "Failed to generate embeddings.",
            error_code="EMBEDDING_ERROR",
            recovery_hint="Check that the embedding model is loaded. Try restarting the application.",
        )


class VectorStoreError(ResearchToolError):
    """Raised when vector store operations fail."""

    def __init__(self, operation: str, message: Optional[str] = None) -> None:
        self.operation = operation
        super().__init__(
            message=message or f"Vector store operation '{operation}' failed.",
            error_code="VECTOR_STORE_ERROR",
            recovery_hint="Try re-processing your URLs to rebuild the index.",
        )


class LLMError(ResearchToolError):
    """Raised when LLM inference fails."""

    def __init__(
        self,
        provider: str,
        message: Optional[str] = None,
    ) -> None:
        self.provider = provider
        super().__init__(
            message=message or f"LLM provider '{provider}' returned an error.",
            error_code="LLM_ERROR",
            recovery_hint="Verify your API key is valid and you have sufficient credits.",
        )


class RateLimitError(ResearchToolError):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        msg = "Rate limit exceeded."
        if retry_after:
            msg += f" Retry after {retry_after:.1f}s."
        super().__init__(
            message=msg,
            error_code="RATE_LIMIT_ERROR",
            recovery_hint="Please wait before making another request.",
        )


class ExportError(ResearchToolError):
    """Raised when export operations fail."""

    def __init__(self, format: str, message: Optional[str] = None) -> None:
        self.format = format
        super().__init__(
            message=message or f"Failed to export in '{format}' format.",
            error_code="EXPORT_ERROR",
            recovery_hint="Ensure you have research data to export. Try a different format.",
        )
