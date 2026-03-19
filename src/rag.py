"""
RAG (Retrieval-Augmented Generation) pipeline with conversation memory.

Integrates the hybrid retriever with LLM generation, maintaining
multi-turn conversation context for follow-up questions.
"""

import os
from typing import Any, Optional

try:
    from langchain.chains import RetrievalQA
except ImportError:
    RetrievalQA = None  # Not used directly in current code

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from src.retriever import HybridRetriever, BM25Retriever, RetrievedChunk
from src.cache import QueryCache
from src.utils import setup_logger, timed, truncate_text
from src.exceptions import LLMError

logger = setup_logger(__name__)

# ─────────────────── Conversation Memory ─────────────────────────────


class ConversationMemory:
    """Sliding-window conversation memory for multi-turn context.

    Maintains the last `max_turns` exchanges (user + assistant) and
    injects them into the RAG prompt so follow-up questions work
    naturally (e.g., "Tell me more about that").

    Args:
        max_turns: Maximum number of Q&A pairs to retain.
    """

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._history: list[dict[str, str]] = []

    def add_exchange(self, question: str, answer: str) -> None:
        """Record a Q&A exchange.

        Args:
            question: User's question.
            answer: Assistant's answer.
        """
        self._history.append({"question": question, "answer": answer})
        # Trim to max_turns
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]

    def get_context_string(self) -> str:
        """Format conversation history as a string for prompt injection.

        Returns:
            Formatted string of previous exchanges, or empty string if none.
        """
        if not self._history:
            return ""

        lines = ["Previous conversation:"]
        for exchange in self._history:
            lines.append(f"User: {exchange['question']}")
            lines.append(f"Assistant: {truncate_text(exchange['answer'], 500)}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        return len(self._history)


# ──────────────────────── LLM Factory ────────────────────────────────

RELEVANCE_THRESHOLD = 0.35

# Prompt with conversation memory support
RAG_PROMPT_TEMPLATE = """You are an expert equity research analyst. Use the following context and conversation history to answer the question.

If the context doesn't contain enough information, say so clearly. Do not fabricate information.
Cite specific details from the sources when possible.

{conversation_history}

Context from retrieved sources:
{context}

Question: {question}

Provide a thorough, well-structured answer:"""


def get_llm(
    api_key: Optional[str] = None,
    provider: str = "openai",
) -> Optional[Any]:
    """Initialize an LLM instance based on provider.

    Args:
        api_key: API key for the provider. If None, returns None.
        provider: LLM provider ('openai' or 'google').

    Returns:
        LLM instance, or None if no API key provided.

    Raises:
        LLMError: If initialization fails with a valid API key.
    """
    if not api_key:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "google":
            api_key = os.environ.get("GOOGLE_API_KEY")
        elif provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    try:
        if provider == "openai":
            return ChatOpenAI(
                api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.3,
            )
        elif provider == "google":
            # Try models in order — different models may have separate quotas
            model = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash")
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model,
                temperature=0.3,
            )
        elif provider == "groq":
            model = os.environ.get("GROQ_MODEL", "llama3-8b-8192")
            return ChatGroq(
                api_key=api_key,
                model_name=model,
                temperature=0.3,
            )
        else:
            raise LLMError(provider, f"Unknown provider: {provider}")
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(provider, str(e))


# ────────────────────── Answer Generation ────────────────────────────

# Module-level cache and memory instances
_query_cache = QueryCache(max_size=128, ttl_seconds=3600)
_conversation_memory = ConversationMemory(max_turns=5)


def get_cache() -> QueryCache:
    """Access the global query cache (for metrics display)."""
    return _query_cache


def get_memory() -> ConversationMemory:
    """Access the global conversation memory."""
    return _conversation_memory


@timed
def get_answer(
    query: str,
    vector_store: Any,
    api_key: Optional[str] = None,
    provider: str = "openai",
    bm25_retriever: Optional[BM25Retriever] = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Generate an answer using the hybrid RAG pipeline.

    Pipeline: Cache check → Hybrid retrieval → LLM generation (if available) → Cache store

    Args:
        query: User's question.
        vector_store: FAISS vector store.
        api_key: Optional LLM API key.
        provider: LLM provider ('openai' or 'google').
        bm25_retriever: Optional BM25 index for hybrid search.
        use_cache: Whether to check/update the cache.

    Returns:
        Dict with keys: answer, sources, chunks_with_scores, has_llm, is_relevant,
        retrieval_method, response_time_ms.
    """
    logger.info(f"get_answer invoked! provider={provider}, api_key_length={len(api_key) if api_key else 0}")

    if not vector_store:
        return {
            "answer": "Vector store not found. Please process URLs first.",
            "sources": [],
            "chunks_with_scores": [],
            "has_llm": False,
            "is_relevant": False,
            "retrieval_method": "none",
        }

    # ─── Step 1: Cache check ───
    if use_cache:
        cached = _query_cache.get(query, provider=provider)
        if cached is not None:
            logger.info("Cache HIT — returning cached answer")
            cached["from_cache"] = True
            return cached

    # ─── Step 2: Hybrid retrieval ───
    use_hybrid = bm25_retriever is not None and bm25_retriever.bm25 is not None
    retrieval_method = "hybrid" if use_hybrid else "faiss_only"

    if use_hybrid:
        hybrid = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            use_reranker=True,
        )
        retrieved_chunks = hybrid.retrieve(query, initial_k=10, final_k=5)
    else:
        # Fallback to FAISS-only
        docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
        retrieved_chunks = []
        for doc, distance in docs_with_scores:
            similarity = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
            retrieved_chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    score=similarity,
                    method="faiss",
                )
            )

    # Deduplicate
    seen_content: set[str] = set()
    unique_chunks: list[RetrievedChunk] = []
    for chunk in retrieved_chunks:
        sig = chunk.text[:100]
        if sig not in seen_content and len(chunk.text.strip()) > 50:
            seen_content.add(sig)
            unique_chunks.append(chunk)

    # Build context
    chunks_with_scores = [
        {"text": c.text, "score": c.score, "source": c.source, "method": c.method}
        for c in unique_chunks
    ]
    source_knowledge = "\n\n".join(c.text for c in unique_chunks)
    sources = list(set(c.source for c in unique_chunks))

    # Relevance check
    max_score = max((c.score for c in unique_chunks), default=0.0)
    is_relevant = max_score >= RELEVANCE_THRESHOLD

    # ─── Step 3: LLM generation ───
    llm = get_llm(api_key, provider)
    conversation_context = _conversation_memory.get_context_string()

    if llm:
        logger.info(f"Generating answer with {provider} (hybrid={use_hybrid})...")
        try:
            prompt = RAG_PROMPT_TEMPLATE.format(
                conversation_history=conversation_context or "No prior conversation.",
                context=truncate_text(source_knowledge, 4000),
                question=query,
            )

            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            if not is_relevant:
                answer = (
                    f"⚠️ **Note:** Retrieved information may not be highly relevant "
                    f"(max score: {max_score:.2%}). Consider processing more relevant URLs.\n\n"
                    + answer
                )

            # Update conversation memory
            _conversation_memory.add_exchange(query, answer)

            result = {
                "answer": answer,
                "sources": sources,
                "context": source_knowledge,
                "chunks_with_scores": chunks_with_scores,
                "has_llm": True,
                "is_relevant": is_relevant,
                "retrieval_method": retrieval_method,
                "from_cache": False,
            }

            # Cache the result
            if use_cache:
                _query_cache.put(query, result, provider=provider)

            return result

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fall through to retrieval-only

    # ─── Step 4: Retrieval-only fallback ───
    logger.info("No LLM available — returning retrieved chunks.")

    formatted_answer = "**Retrieval Only (No LLM Configured):**\n\n"

    if not is_relevant:
        formatted_answer += (
            f"⚠️ **Warning:** Retrieved information may not be highly relevant "
            f"(max score: {max_score:.2%}).\n\n"
        )

    formatted_answer += "**Based on the retrieved information:**\n\n"
    formatted_answer += truncate_text(source_knowledge, 2000)

    _conversation_memory.add_exchange(query, formatted_answer)

    result = {
        "answer": formatted_answer,
        "sources": sources,
        "context": source_knowledge,
        "chunks_with_scores": chunks_with_scores,
        "has_llm": False,
        "is_relevant": is_relevant,
        "retrieval_method": retrieval_method,
        "from_cache": False,
    }

    if use_cache:
        _query_cache.put(query, result, provider=provider)

    return result
