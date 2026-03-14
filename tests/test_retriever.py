"""
Tests for src/retriever.py — BM25, FAISS, RRF fusion, and re-ranking.
"""

import pytest
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.retriever import (
    BM25Retriever,
    RetrievedChunk,
    reciprocal_rank_fusion,
)


class TestBM25Retriever:
    """Tests for BM25 keyword search."""

    def test_build_index(self, sample_chunks):
        retriever = BM25Retriever()
        retriever.build_index(sample_chunks)
        assert retriever.bm25 is not None
        assert len(retriever.documents) == len(sample_chunks)

    def test_retrieve_relevant(self, bm25_retriever):
        results = bm25_retriever.retrieve("Apple revenue", k=3)
        assert len(results) > 0
        assert any("Apple" in r.text for r in results)

    def test_retrieve_returns_scored_chunks(self, bm25_retriever):
        results = bm25_retriever.retrieve("Tesla delivery", k=3)
        for result in results:
            assert isinstance(result, RetrievedChunk)
            assert result.score > 0
            assert result.method == "bm25"

    def test_retrieve_empty_index(self):
        retriever = BM25Retriever()
        results = retriever.retrieve("anything")
        assert results == []

    def test_top_result_is_most_relevant(self, bm25_retriever):
        results = bm25_retriever.retrieve("Google Gemini AI", k=3)
        if len(results) > 1:
            assert results[0].score >= results[1].score


class TestReciprocalRankFusion:
    """Tests for RRF fusion algorithm."""

    def test_fuses_two_lists(self, sample_retrieved_chunks):
        list1 = sample_retrieved_chunks[:2]
        list2 = sample_retrieved_chunks[1:]

        fused = reciprocal_rank_fusion([list1, list2])
        assert len(fused) > 0

    def test_all_results_have_hybrid_method(self, sample_retrieved_chunks):
        fused = reciprocal_rank_fusion([sample_retrieved_chunks])
        for chunk in fused:
            assert chunk.method == "hybrid"

    def test_higher_ranked_items_score_higher(self):
        list1 = [
            RetrievedChunk(text="A" * 250, source="s1", score=0.9, method="bm25"),
            RetrievedChunk(text="B" * 250, source="s2", score=0.5, method="bm25"),
        ]
        list2 = [
            RetrievedChunk(text="A" * 250, source="s1", score=0.8, method="faiss"),
            RetrievedChunk(text="C" * 250, source="s3", score=0.4, method="faiss"),
        ]

        fused = reciprocal_rank_fusion([list1, list2])
        # "A" appears in both lists, should be ranked first
        assert "A" in fused[0].text

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []

    def test_single_list(self, sample_retrieved_chunks):
        fused = reciprocal_rank_fusion([sample_retrieved_chunks])
        assert len(fused) == len(sample_retrieved_chunks)


class TestBM25SaveLoad:
    """Tests for BM25 index persistence."""

    def test_save_and_load(self, bm25_retriever, tmp_path):
        path = str(tmp_path / "test_bm25.pkl")
        bm25_retriever.save(path)

        loaded = BM25Retriever()
        assert loaded.load(path) is True
        assert len(loaded.documents) == len(bm25_retriever.documents)

    def test_load_nonexistent(self):
        retriever = BM25Retriever()
        assert retriever.load("nonexistent.pkl") is False
