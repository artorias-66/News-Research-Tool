"""
Tests for the FastAPI backend — endpoints, rate limiting, and error handling.

Uses pytest-asyncio with httpx.AsyncClient for testing async FastAPI endpoints.
"""

import pytest
import httpx
from unittest.mock import MagicMock

from api.main import app, _state
from src.retriever import BM25Retriever


# ─── Helper ──────────────────────────────────────────────────────────

async def _make_client():
    """Create an AsyncClient with the app transport."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ─── Health ──────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    @pytest.mark.asyncio
    async def test_returns_200(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/health")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_structure(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/health")
            data = response.json()
            assert "status" in data
            assert "vector_store_loaded" in data
            assert "bm25_loaded" in data


# ─── Metrics ─────────────────────────────────────────────────────────


class TestMetricsEndpoint:
    """Tests for GET /api/metrics."""

    @pytest.mark.asyncio
    async def test_returns_200(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/metrics")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_has_cache_metrics(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/metrics")
            data = response.json()
            assert "cache" in data
            assert "hits" in data["cache"]


# ─── Query ───────────────────────────────────────────────────────────


class TestQueryEndpoint:
    """Tests for POST /api/query."""

    @pytest.mark.asyncio
    async def test_no_index_returns_400(self):
        _state["vector_store"] = None
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/query",
                json={"question": "test question"},
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_query_with_mock(self):
        mock_vs = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = (
            "Apple Inc. reported strong earnings with $94.8 billion in revenue. "
            "iPhone 16 sales exceeded expectations in the quarter."
        )
        mock_doc.metadata = {"source": "https://example.com"}
        mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.5)]
        _state["vector_store"] = mock_vs
        _state["bm25_retriever"] = BM25Retriever()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/query",
                json={"question": "Apple earnings"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data

        _state["vector_store"] = None


# ─── Export ──────────────────────────────────────────────────────────


class TestExportEndpoint:
    """Tests for POST /api/export."""

    @pytest.mark.asyncio
    async def test_json_export(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/export",
                json={
                    "format": "json",
                    "messages": [
                        {"role": "user", "content": "Q1"},
                        {"role": "assistant", "content": "A1"},
                    ],
                },
            )
            assert response.status_code == 200
            assert response.json()["format"] == "json"

    @pytest.mark.asyncio
    async def test_csv_export(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/export",
                json={
                    "format": "csv",
                    "messages": [
                        {"role": "user", "content": "Q1"},
                        {"role": "assistant", "content": "A1"},
                    ],
                },
            )
            assert response.status_code == 200
            assert response.json()["format"] == "csv"

    @pytest.mark.asyncio
    async def test_invalid_format(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/api/export",
                json={"format": "xml", "messages": []},
            )
            assert response.status_code == 400
