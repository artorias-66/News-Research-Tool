"""
Tests for src/utils.py — text cleaning, URL validation, hashing, timing.
"""

import pytest

from src.utils import clean_text, validate_url, hash_content, truncate_text, timed
from src.exceptions import URLValidationError


class TestCleanText:
    """Tests for the clean_text function."""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_removes_noise_patterns(self):
        text = "Important news content. Tags Share Tweet Follow Subscribe"
        result = clean_text(text)
        assert "Tags" not in result
        assert "Share" not in result
        assert "Important news content." in result

    def test_removes_short_lines(self):
        text = "Nav\nMenu\nThis is a proper sentence with enough content."
        result = clean_text(text)
        assert "Nav" not in result
        assert "This is a proper sentence" in result

    def test_preserves_sentences_ending_with_period(self):
        text = "Short."
        result = clean_text(text)
        assert "Short." in result

    def test_preserves_questions(self):
        text = "Why?"
        result = clean_text(text)
        assert "Why?" in result

    def test_collapses_whitespace(self):
        text = "Hello     world.\n\n\n\nThis is a long sentence with enough chars."
        result = clean_text(text)
        assert "     " not in result
        assert "\n\n\n" not in result


class TestValidateUrl:
    """Tests for the validate_url function."""

    def test_valid_https_url(self):
        assert validate_url("https://example.com/article") == "https://example.com/article"

    def test_valid_http_url(self):
        assert validate_url("http://example.com") == "http://example.com"

    def test_strips_whitespace(self):
        assert validate_url("  https://example.com  ") == "https://example.com"

    def test_empty_url_raises(self):
        with pytest.raises(URLValidationError):
            validate_url("")

    def test_missing_scheme_raises(self):
        with pytest.raises(URLValidationError):
            validate_url("ftp://example.com")

    def test_missing_domain_raises(self):
        with pytest.raises(URLValidationError):
            validate_url("https://")

    def test_invalid_domain_raises(self):
        with pytest.raises(URLValidationError):
            validate_url("https://localhost")


class TestHashContent:
    """Tests for the hash_content function."""

    def test_deterministic(self):
        assert hash_content("hello") == hash_content("hello")

    def test_different_inputs(self):
        assert hash_content("hello") != hash_content("world")

    def test_returns_hex_string(self):
        result = hash_content("test")
        assert all(c in "0123456789abcdef" for c in result)


class TestTruncateText:
    """Tests for the truncate_text function."""

    def test_short_text_unchanged(self):
        assert truncate_text("hello", 100) == "hello"

    def test_long_text_truncated(self):
        result = truncate_text("a" * 100, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_custom_suffix(self):
        result = truncate_text("a" * 100, 50, suffix="[more]")
        assert result.endswith("[more]")


class TestTimed:
    """Tests for the timed decorator."""

    def test_returns_result(self):
        @timed
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_tracks_elapsed(self):
        @timed
        def slow_func():
            import time
            time.sleep(0.05)
            return "done"

        slow_func()
        assert slow_func._last_elapsed > 0.04
