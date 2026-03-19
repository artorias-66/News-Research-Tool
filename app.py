"""
Equity Research Tool — Streamlit Frontend.

Provides a chat-based research interface backed by the hybrid RAG pipeline.
Features: dynamic URL ingestion, conversation memory, analytics dashboard,
and research export.
"""

# ─── Python 3.14 Compatibility Patch ─────────────────────────────────
# Altair uses TypedDict(closed=True) which crashes on Python 3.14 due to
# metaclass changes. Patch _TypedDictMeta.__new__ before importing anything.
import sys
if sys.version_info >= (3, 14):
    import typing
    _orig_new = typing._TypedDictMeta.__new__

    def _patched_new(cls, name, bases, ns, *, closed=False, total=True, **kwargs):
        return _orig_new(cls, name, bases, ns, total=total, **kwargs)
    typing._TypedDictMeta.__new__ = _patched_new
# ─── End Patch ────────────────────────────────────────────────────────

import time
import streamlit as st
from dotenv import load_dotenv; load_dotenv()  # noqa: E702

from src.ui import load_css, render_header, render_sidebar, render_analytics, render_export_controls
from src.ingest import load_urls, process_documents
from src.vector_store import create_vector_store, load_vector_store, load_index_metadata
from src.retriever import BM25Retriever
from src.rag import get_answer, get_cache, get_memory
from src.utils import setup_logger

logger = setup_logger(__name__)

# ─────────────────── Page Config ─────────────────────────────────────

st.set_page_config(page_title="Equity Research Tool", page_icon="📈", layout="wide")
load_css("assets/style.css")


def main() -> None:
    render_header()

    # Sidebar (returns chunk_size and chunk_overlap now)
    urls, process_button, chunk_size, chunk_overlap = render_sidebar()

    # ─── Initialize Session State ───
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store()
    if "bm25_retriever" not in st.session_state:
        bm25 = BM25Retriever()
        bm25.load()
        st.session_state.bm25_retriever = bm25
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "response_times" not in st.session_state:
        st.session_state.response_times = []

    # ─── Process URLs ───
    if process_button:
        if not urls:
            st.error("Please enter at least one URL.")
        else:
            with st.status("Processing URLs...", expanded=True) as status:
                try:
                    st.write("📥 Fetching content from URLs (async with retry)...")
                    documents = load_urls(urls)

                    st.write("✂️ Splitting text into chunks...")
                    chunks = process_documents(documents, chunk_size, chunk_overlap)
                    st.info(f"📊 Created {len(chunks)} chunks from {len(documents)} URL(s)")

                    st.write("🧠 Building FAISS vector index...")
                    vector_store = create_vector_store(chunks)
                    st.session_state.vector_store = vector_store

                    st.write("📝 Building BM25 keyword index...")
                    bm25 = BM25Retriever()
                    bm25.build_index(chunks)
                    bm25.save()
                    st.session_state.bm25_retriever = bm25

                    st.session_state.sources = urls

                    status.update(
                        label="✅ Processing Complete! Both indices ready.",
                        state="complete",
                        expanded=False,
                    )
                    st.balloons()

                except Exception as e:
                    status.update(label="❌ Error Occurred", state="error")
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error processing URLs: {e}", exc_info=True)

    # ─── Tabs: Chat | Analytics ───
    chat_tab, analytics_tab = st.tabs(["💬 Research Chat", "📊 Analytics"])

    with chat_tab:
        # Conversation controls
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                get_memory().clear()
                st.rerun()

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about the analyzed articles..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching with hybrid retrieval..."):
                    if st.session_state.vector_store:
                        start_time = time.perf_counter()

                        response = get_answer(
                            query=prompt,
                            vector_store=st.session_state.vector_store,
                            bm25_retriever=st.session_state.bm25_retriever,
                        )

                        elapsed = time.perf_counter() - start_time
                        st.session_state.response_times.append(elapsed)

                        answer = response["answer"]
                        sources = response["sources"]
                        chunks_with_scores = response.get("chunks_with_scores", [])
                        retrieval_method = response.get("retrieval_method", "unknown")
                        from_cache = response.get("from_cache", False)

                        st.markdown(answer)

                        # Show retrieval info badge
                        badge_parts = [f"⏱️ {elapsed:.2f}s", f"🔍 {retrieval_method}"]
                        if from_cache:
                            badge_parts.append("⚡ cached")
                        st.caption(" · ".join(badge_parts))

                        # Source sections
                        if chunks_with_scores:
                            with st.expander("🔍 View Retrieved Chunks"):
                                for rank, chunk_data in enumerate(chunks_with_scores, 1):
                                    score = chunk_data["score"]
                                    method = chunk_data.get("method", "unknown")
                                    st.markdown(
                                        f"**Chunk {rank}** (Score: {score:.4f} | Method: {method})"
                                    )
                                    st.markdown(f"*Source: {chunk_data['source']}*")
                                    st.write(chunk_data["text"][:500])
                                    st.markdown("---")

                        if sources:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"{i}. [{source}]({source})")

                        # Save to history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                    else:
                        st.warning("⚠️ Please process some URLs first!")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": "⚠️ Please process URLs first!"}
                        )

    with analytics_tab:
        cache_metrics = get_cache().metrics.to_dict()
        index_meta = load_index_metadata()
        render_analytics(
            metrics=cache_metrics,
            index_metadata=index_meta,
            response_times=st.session_state.get("response_times", []),
        )

    # Export controls
    render_export_controls(
        messages=st.session_state.messages,
        sources=st.session_state.get("sources", []),
    )


if __name__ == "__main__":
    main()
