"""
Shared test fixtures for pytest.
"""

import pytest
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.cache import QueryCache
from src.retriever import BM25Retriever, RetrievedChunk


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create a small set of test documents."""
    return [
        Document(
            page_content="Apple Inc. reported record revenue of $94.8 billion for Q4 2024. "
            "iPhone 16 sales exceeded analyst expectations by 15%. The company's "
            "services division also showed strong growth, reaching $22 billion in revenue. "
            "CEO Tim Cook attributed the success to strong demand in emerging markets.",
            metadata={"source": "https://example.com/apple-earnings"},
        ),
        Document(
            page_content="Google announced Gemini 2.0, a next-generation multimodal AI model. "
            "The model demonstrates significant improvements in reasoning, code generation, "
            "and visual understanding compared to its predecessor. Sundar Pichai called it "
            "a breakthrough in artificial intelligence research and development.",
            metadata={"source": "https://example.com/google-gemini"},
        ),
        Document(
            page_content="Tesla's stock surged 12% after the company reported better-than-expected "
            "delivery numbers for Q4 2024. The company delivered 500,000 vehicles globally, "
            "led by strong Model Y demand. Analysts raised their price targets following "
            "the news, citing improved production efficiency at the Berlin Gigafactory.",
            metadata={"source": "https://example.com/tesla-deliveries"},
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents: list[Document]) -> list[Document]:
    """Create pre-processed text chunks from sample documents."""
    # Simulate chunking — keep them as-is since they're already chunk-sized
    return sample_documents


@pytest.fixture
def sample_retrieved_chunks() -> list[RetrievedChunk]:
    """Create sample retrieval results for testing."""
    return [
        RetrievedChunk(
            text="Apple reported record revenue of $94.8 billion.",
            source="https://example.com/apple",
            score=0.85,
            method="faiss",
        ),
        RetrievedChunk(
            text="Google announced Gemini 2.0 AI model.",
            source="https://example.com/google",
            score=0.72,
            method="bm25",
        ),
        RetrievedChunk(
            text="Tesla delivered 500,000 vehicles in Q4.",
            source="https://example.com/tesla",
            score=0.65,
            method="faiss",
        ),
    ]


@pytest.fixture
def bm25_retriever(sample_chunks: list[Document]) -> BM25Retriever:
    """Create a BM25 retriever with sample data indexed."""
    retriever = BM25Retriever()
    retriever.build_index(sample_chunks)
    return retriever


@pytest.fixture
def query_cache() -> QueryCache:
    """Create a small cache for testing."""
    return QueryCache(max_size=5, ttl_seconds=10.0)


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Create sample chat messages for export testing."""
    return [
        {"role": "user", "content": "What were Apple's Q4 earnings?"},
        {"role": "assistant", "content": "Apple reported record revenue of $94.8 billion."},
        {"role": "user", "content": "How did Tesla perform?"},
        {"role": "assistant", "content": "Tesla delivered 500,000 vehicles in Q4 2024."},
    ]
