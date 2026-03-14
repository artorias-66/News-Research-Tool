"""
Tests for src/ingest.py — URL fetching, document processing, and retry logic.
"""

import pytest
from unittest.mock import patch, MagicMock

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.ingest import (
    extract_text_from_html,
    process_documents,
    _fetch_url_sync,
    load_urls,
)
from src.exceptions import URLFetchError, URLValidationError


class TestExtractTextFromHtml:
    """Tests for HTML text extraction."""

    def test_extracts_body_text(self):
        html = "<html><body><p>Hello world, this is a paragraph with enough content.</p></body></html>"
        result = extract_text_from_html(html)
        assert "Hello world" in result

    def test_removes_script_tags(self):
        html = "<html><body><script>var x = 1;</script><p>Visible content here for testing.</p></body></html>"
        result = extract_text_from_html(html)
        assert "var x = 1" not in result
        assert "Visible content" in result

    def test_removes_nav_and_footer(self):
        html = "<html><body><nav>Menu items</nav><p>Main article content goes here.</p><footer>Copyright</footer></body></html>"
        result = extract_text_from_html(html)
        assert "Menu items" not in result
        assert "Copyright" not in result
        assert "Main article content" in result

    def test_handles_empty_html(self):
        assert extract_text_from_html("") == ""


class TestProcessDocuments:
    """Tests for document chunking."""

    def test_creates_chunks(self, sample_documents):
        chunks = process_documents(sample_documents, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 0

    def test_respects_chunk_size(self, sample_documents):
        chunk_size = 300
        chunks = process_documents(sample_documents, chunk_size=chunk_size, chunk_overlap=50)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size * 1.2  # Allow small overflow from splitter

    def test_preserves_metadata(self, sample_documents):
        chunks = process_documents(sample_documents, chunk_size=200, chunk_overlap=50)
        for chunk in chunks:
            assert "source" in chunk.metadata

    def test_filters_tiny_chunks(self):
        docs = [Document(page_content="x" * 30, metadata={"source": "test"})]
        chunks = process_documents(docs, chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 0  # Too short (< 50 chars after cleaning)


class TestFetchUrlSync:
    """Tests for synchronous URL fetching with retry."""

    @patch("src.ingest.requests.get")
    def test_successful_fetch(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>This is a test article with sufficient content to pass the length check.</p></body></html>"
        mock_get.return_value = mock_response

        doc = _fetch_url_sync("https://example.com/article")
        assert isinstance(doc, Document)
        assert "test article" in doc.page_content

    @patch("src.ingest.requests.get")
    def test_404_retries_and_fails(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(URLFetchError):
            _fetch_url_sync("https://example.com/missing")

    @patch("src.ingest.requests.get")
    def test_retries_on_exception(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")

        with pytest.raises(URLFetchError):
            _fetch_url_sync("https://example.com/down")

        assert mock_get.call_count == 3  # MAX_RETRIES


class TestLoadUrls:
    """Tests for the main URL loading function."""

    def test_validates_urls(self):
        with pytest.raises(URLValidationError):
            load_urls(["not-a-url"])

    def test_deduplicates_urls(self):
        # Two URLs that should deduplicate (trailing slash normalization)
        with patch("src.ingest._fetch_url_sync") as mock_fetch, \
             patch("src.ingest.asyncio") as mock_asyncio:
            # Force sync path by simulating running event loop
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_asyncio.get_running_loop.return_value = mock_loop

            mock_fetch.return_value = Document(
                page_content="Test content " * 20,
                metadata={"source": "test"},
            )
            result = load_urls([
                "https://example.com/article",
                "https://example.com/article/",
            ])
            # Should only fetch once (dedup strips trailing slash)
            assert mock_fetch.call_count == 1

