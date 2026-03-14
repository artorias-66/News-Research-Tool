"""
FAISS vector store management.

Handles creation, persistence, and loading of the FAISS index
along with metadata tracking for analytics.
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.utils import setup_logger
from src.exceptions import VectorStoreError, EmbeddingError

logger = setup_logger(__name__)

# ───────────────────────── Constants ─────────────────────────────────

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
METADATA_PATH = os.path.join(FAISS_INDEX_PATH, "index_metadata.json")


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings instance.

    Raises:
        EmbeddingError: If the model fails to load.
    """
    try:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        return HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        raise EmbeddingError(f"Failed to load embedding model '{MODEL_NAME}': {e}")


def _save_metadata(
    chunk_count: int,
    source_urls: list[str],
) -> None:
    """Save index metadata for analytics.

    Args:
        chunk_count: Number of chunks indexed.
        source_urls: List of source URLs processed.
    """
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": chunk_count,
        "source_count": len(source_urls),
        "sources": source_urls,
        "embedding_model": MODEL_NAME,
    }
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Index metadata saved: {chunk_count} chunks, {len(source_urls)} sources")


def load_index_metadata() -> Optional[dict]:
    """Load index metadata for analytics display.

    Returns:
        Metadata dict, or None if no metadata file exists.
    """
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def create_vector_store(chunks: list[Document]) -> FAISS:
    """Create and persist a FAISS vector store from document chunks.

    Args:
        chunks: List of LangChain Document chunks to embed and index.

    Returns:
        FAISS vector store instance.

    Raises:
        VectorStoreError: If index creation or saving fails.
        EmbeddingError: If embedding generation fails.
    """
    try:
        embeddings = get_embeddings()
        logger.info(f"Creating FAISS index with {len(chunks)} chunks...")
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Save index
        vector_store.save_local(FAISS_INDEX_PATH)
        logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")

        # Save metadata
        source_urls = list(set(
            c.metadata.get("source", "Unknown") for c in chunks
        ))
        _save_metadata(chunk_count=len(chunks), source_urls=source_urls)

        return vector_store
    except EmbeddingError:
        raise
    except Exception as e:
        raise VectorStoreError("create", str(e))


def load_vector_store() -> Optional[FAISS]:
    """Load existing FAISS vector store from disk.

    Returns:
        FAISS vector store if found, None otherwise.

    Raises:
        VectorStoreError: If index loading fails (corrupted, etc.).
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning("No FAISS index found.")
        return None

    try:
        embeddings = get_embeddings()
        logger.info("Loading FAISS index...")

        try:
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except TypeError:
            # Fallback for older FAISS versions
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

        return vector_store
    except EmbeddingError:
        raise
    except Exception as e:
        raise VectorStoreError("load", str(e))
