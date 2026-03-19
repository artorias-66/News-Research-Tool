"""
Tests for src/rag.py — RAG pipeline, conversation memory, and answer generation.
"""

from unittest.mock import patch, MagicMock

from src.rag import ConversationMemory, get_answer


class TestConversationMemory:
    """Tests for the sliding-window conversation memory."""

    def test_add_exchange(self):
        memory = ConversationMemory(max_turns=3)
        memory.add_exchange("What is AI?", "AI stands for Artificial Intelligence.")
        assert memory.turn_count == 1

    def test_sliding_window(self):
        memory = ConversationMemory(max_turns=2)
        memory.add_exchange("Q1", "A1")
        memory.add_exchange("Q2", "A2")
        memory.add_exchange("Q3", "A3")
        assert memory.turn_count == 2  # Oldest trimmed

    def test_context_string(self):
        memory = ConversationMemory()
        memory.add_exchange("What is FAISS?", "FAISS is a vector similarity search library.")
        context = memory.get_context_string()
        assert "FAISS" in context
        assert "Previous conversation:" in context

    def test_empty_context(self):
        memory = ConversationMemory()
        assert memory.get_context_string() == ""

    def test_clear(self):
        memory = ConversationMemory()
        memory.add_exchange("Q", "A")
        memory.clear()
        assert memory.turn_count == 0


class TestGetAnswer:
    """Tests for the RAG answer generation pipeline."""

    def test_no_vector_store(self):
        result = get_answer("test", vector_store=None, use_cache=False)
        assert "process URLs" in result["answer"].lower() or "vector store" in result["answer"].lower()
        assert result["has_llm"] is False

    def test_retrieval_only_mode(self):
        """Test with a mocked vector store but no LLM."""
        mock_vs = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = (
            "Apple Inc. reported $94.8 billion in revenue for Q4 2024. "
            "iPhone 16 sales exceeded analyst expectations by a wide margin."
        )
        mock_doc.metadata = {"source": "https://example.com"}
        mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.5)]

        result = get_answer(
            "Apple earnings",
            vector_store=mock_vs,
            use_cache=False,
        )

        assert result["has_llm"] is False
        assert len(result["sources"]) > 0
        assert result["retrieval_method"] == "faiss_only"

    @patch("src.rag.get_llm")
    def test_llm_generation(self, mock_get_llm):
        """Test with mocked LLM for generation."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Apple had strong Q4 earnings with $94.8B revenue."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        mock_vs = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = (
            "Apple Inc. reported revenue of $94.8 billion for Q4 2024. "
            "iPhone sales exceeded analyst expectations with strong performance in emerging markets."
        )
        mock_doc.metadata = {"source": "https://example.com"}
        mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.3)]

        result = get_answer(
            "Apple earnings",
            vector_store=mock_vs,
            use_cache=False,
        )

        assert result["has_llm"] is True
        assert "94.8" in result["answer"]
