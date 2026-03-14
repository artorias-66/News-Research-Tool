"""
Hybrid retrieval pipeline: BM25 + FAISS + Reciprocal Rank Fusion + Cross-Encoder Re-ranker.

This is the core differentiator of the project. Combines keyword-based (BM25)
and semantic (FAISS) retrieval, fuses results via RRF, then re-ranks with a
cross-encoder for maximum precision.

Architecture:
    Query → [BM25 Retriever] ─┐
                               ├─→ RRF Fusion → Cross-Encoder Re-rank → Top-K
    Query → [FAISS Retriever] ─┘
"""

import pickle
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from rank_bm25 import BM25Okapi
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from sentence_transformers import CrossEncoder

from src.utils import setup_logger, timed

logger = setup_logger(__name__)

# ─────────────────────────── Data Types ──────────────────────────────

@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata.

    Attributes:
        text: The chunk content.
        source: URL or document source.
        score: Retrieval/re-ranking score (higher = more relevant).
        method: Which retriever produced this result ('bm25', 'faiss', 'hybrid', 'reranked').
    """

    text: str
    source: str
    score: float
    method: str = "unknown"


# ─────────────────────── BM25 Retriever ──────────────────────────────

BM25_INDEX_PATH = "bm25_index.pkl"


class BM25Retriever:
    """BM25Okapi keyword-based retriever.

    Uses term frequency and inverse document frequency for
    keyword matching — effective for exact term lookups that
    semantic search may miss.
    """

    def __init__(self) -> None:
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list[Document] = []

    def build_index(self, documents: list[Document]) -> None:
        """Build a BM25 index from document chunks.

        Args:
            documents: List of LangChain Documents to index.
        """
        self.documents = documents
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(documents)} documents")

    def save(self, path: str = BM25_INDEX_PATH) -> None:
        """Persist BM25 index to disk.

        Args:
            path: File path for the saved index.
        """
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: str = BM25_INDEX_PATH) -> bool:
        """Load a BM25 index from disk.

        Args:
            path: File path to load from.

        Returns:
            True if loaded successfully, False if file not found.
        """
        if not os.path.exists(path):
            logger.warning("No BM25 index found.")
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
        logger.info(f"BM25 index loaded from {path} ({len(self.documents)} documents)")
        return True

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Retrieve top-k documents by BM25 score.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by BM25 score (descending).
        """
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                doc = self.documents[idx]
                results.append(
                    RetrievedChunk(
                        text=doc.page_content,
                        source=doc.metadata.get("source", "Unknown"),
                        score=float(scores[idx]),
                        method="bm25",
                    )
                )
        return results


# ─────────────────────── FAISS Retriever ─────────────────────────────


class FAISSRetriever:
    """FAISS-based semantic similarity retriever.

    Uses dense vector embeddings for semantic matching — captures
    meaning and paraphrases that keyword search misses.
    """

    def __init__(self, vector_store=None) -> None:
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Retrieve top-k documents by semantic similarity.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk sorted by similarity score (descending).
        """
        if self.vector_store is None:
            return []

        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)

        results = []
        for doc, distance in docs_with_scores:
            # Convert L2 distance to similarity: 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
            results.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    score=similarity,
                    method="faiss",
                )
            )
        return results


# ──────────────────── Reciprocal Rank Fusion ─────────────────────────


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ 1 / (k + rank_i(d))

    This method is robust to score distribution differences between
    retrievers and is used in production at major search engines.

    Args:
        result_lists: List of ranked result lists from different retrievers.
        k: Smoothing constant (default 60, as per original RRF paper).

    Returns:
        Fused list of RetrievedChunk sorted by combined RRF score.
    """
    # Map: chunk_text → (best_chunk_obj, cumulative_rrf_score)
    fused_scores: dict[str, tuple[RetrievedChunk, float]] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed
            key = chunk.text[:200]  # Use first 200 chars as signature

            if key in fused_scores:
                existing_chunk, existing_score = fused_scores[key]
                fused_scores[key] = (existing_chunk, existing_score + rrf_score)
            else:
                fused_scores[key] = (chunk, rrf_score)

    # Sort by fused score descending
    sorted_results = sorted(fused_scores.values(), key=lambda x: x[1], reverse=True)

    return [
        RetrievedChunk(
            text=chunk.text,
            source=chunk.source,
            score=score,
            method="hybrid",
        )
        for chunk, score in sorted_results
    ]


# ──────────────────── Cross-Encoder Re-ranker ────────────────────────


class CrossEncoderReranker:
    """Re-rank candidates using a cross-encoder model.

    Cross-encoders jointly encode (query, document) pairs and produce
    a single relevance score — more accurate than bi-encoder similarity
    but computationally heavier, so we only re-rank the top candidates.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading cross-encoder: {self.MODEL_NAME}")
            self._model = CrossEncoder(self.MODEL_NAME)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Re-rank chunks by cross-encoder relevance score.

        Args:
            query: The user's search query.
            chunks: Candidate chunks to re-rank.
            top_k: Number of top results to return.

        Returns:
            Re-ranked list of RetrievedChunk (top_k results).
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                score=float(score),
                method="reranked",
            )
            for chunk, score in scored_chunks[:top_k]
        ]


# ────────────────────── Hybrid Retriever ─────────────────────────────


class HybridRetriever:
    """Full hybrid retrieval pipeline.

    Pipeline: Query → BM25 + FAISS (parallel) → RRF Fusion → Cross-Encoder Re-rank → Top-K

    This implements the same retrieval pattern used in production
    search systems at Google, Meta, and Microsoft.
    """

    def __init__(
        self,
        vector_store=None,
        bm25_retriever: Optional[BM25Retriever] = None,
        use_reranker: bool = True,
    ) -> None:
        self.faiss_retriever = FAISSRetriever(vector_store)
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = CrossEncoderReranker() if use_reranker else None

    @timed
    def retrieve(
        self,
        query: str,
        initial_k: int = 10,
        final_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Execute the full hybrid retrieval pipeline.

        Args:
            query: User's search query.
            initial_k: Number of candidates from each retriever.
            final_k: Number of final results after re-ranking.

        Returns:
            Top-K re-ranked RetrievedChunk results.
        """
        # Step 1: Parallel retrieval from BM25 + FAISS
        bm25_results = self.bm25_retriever.retrieve(query, k=initial_k)
        faiss_results = self.faiss_retriever.retrieve(query, k=initial_k)

        logger.info(
            f"Retrieved {len(bm25_results)} BM25 + {len(faiss_results)} FAISS results"
        )

        # Handle edge cases
        if not bm25_results and not faiss_results:
            return []
        if not bm25_results:
            candidates = faiss_results
        elif not faiss_results:
            candidates = bm25_results
        else:
            # Step 2: RRF Fusion
            candidates = reciprocal_rank_fusion([bm25_results, faiss_results])

        # Step 3: Cross-encoder re-ranking
        if self.reranker and len(candidates) > 1:
            results = self.reranker.rerank(query, candidates, top_k=final_k)
            logger.info(f"Re-ranked to {len(results)} results")
            return results

        return candidates[:final_k]
