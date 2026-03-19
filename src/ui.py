"""
Streamlit UI components for the Equity Research Tool.

Provides sidebar configuration with dynamic URL inputs, advanced settings,
export controls, and analytics display.
"""

import streamlit as st
from typing import Optional


def load_css(file_path: str) -> None:
    """Load and inject custom CSS into Streamlit.

    Args:
        file_path: Path to the CSS file.
    """
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Render the application header with description."""
    st.markdown(
        """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">Equity Research Tool 📈</h1>
        <p style="font-size: 1.1rem; color: #8b949e;">
            Hybrid RAG Pipeline &bull; BM25 + FAISS + Cross-Encoder Re-ranking
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple:
    """Render the sidebar with configuration, dynamic URLs, and advanced settings.

    Returns:
        Tuple of (urls, process_button, chunk_size, chunk_overlap).
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.markdown("---")

        # Dynamic URL Input
        st.header("📰 News Sources")
        st.info("✅ FREE local embeddings — No API needed for retrieval!")

        if "url_count" not in st.session_state:
            st.session_state.url_count = 3

        urls: list[str] = []
        for i in range(st.session_state.url_count):
            url = st.text_input(
                f"URL {i + 1}",
                key=f"url_{i}",
                placeholder="https://example.com/article",
            )
            if url and url.strip():
                urls.append(url.strip())

        # Add / Remove URL buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add URL", use_container_width=True):
                st.session_state.url_count += 1
                st.rerun()
        with col2:
            if st.button("➖ Remove", use_container_width=True) and st.session_state.url_count > 1:
                st.session_state.url_count -= 1
                st.rerun()

        process_button = st.button(
            "🚀 Process URLs", type="primary", use_container_width=True
        )

        st.markdown("---")

        # Advanced Settings
        with st.expander("⚡ Advanced Settings"):
            chunk_size = st.slider(
                "Chunk Size",
                min_value=200,
                max_value=3000,
                value=1000,
                step=100,
                help="Characters per text chunk. Larger = more context, slower retrieval.",
            )
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between adjacent chunks. Prevents context loss at boundaries.",
            )

        st.markdown("---")

        # Feature list
        st.markdown(
            """
        ### 🏗️ Architecture
        - 🔍 **Hybrid Retrieval**: BM25 + FAISS + RRF Fusion
        - 🎯 **Cross-Encoder Re-ranking**: Precision re-scoring
        - 🧠 **Conversation Memory**: Multi-turn follow-ups
        - ⚡ **LRU Cache**: TTL-based query caching
        - 📊 **Export**: JSON, CSV, Research Report
        """
        )

        return urls, process_button, chunk_size, chunk_overlap


def render_analytics(
    metrics: dict,
    index_metadata: Optional[dict] = None,
    response_times: Optional[list[float]] = None,
) -> None:
    """Render the analytics dashboard tab.

    Args:
        metrics: Cache metrics dict from QueryCache.
        index_metadata: Index metadata from vector_store.
        response_times: List of recent response times in seconds.
    """
    st.markdown("### 📊 Analytics Dashboard")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        chunks = index_metadata.get("chunk_count", 0) if index_metadata else 0
        st.metric("Chunks Indexed", chunks)
    with col2:
        sources = index_metadata.get("source_count", 0) if index_metadata else 0
        st.metric("Sources", sources)
    with col3:
        st.metric("Cache Hit Rate", metrics.get("hit_rate", "0.0%"))
    with col4:
        st.metric("Cache Size", metrics.get("total_requests", 0))

    # Response time chart (text-based to avoid altair compatibility issues)
    if response_times and len(response_times) > 1:
        st.markdown("#### ⏱️ Response Times")
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Avg", f"{avg_time:.2f}s")
        with col_b:
            st.metric("Min", f"{min_time:.2f}s")
        with col_c:
            st.metric("Max", f"{max_time:.2f}s")
        # Show recent response times as a simple bar
        st.markdown("**Recent queries:**")
        for i, t in enumerate(response_times[-5:], 1):
            bar_len = int(min(t / max_time, 1.0) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            st.text(f"  Q{len(response_times) - 5 + i}: {bar} {t:.2f}s")


def render_export_controls(
    messages: list[dict[str, str]],
    sources: list[str],
) -> None:
    """Render export buttons in the sidebar.

    Args:
        messages: Chat messages for export.
        sources: List of source URLs.
    """
    from src.export import export_to_json, export_to_csv, generate_report

    if not messages:
        return

    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export")

    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        json_data = export_to_json(messages)
        st.download_button(
            "JSON",
            json_data,
            "research_export.json",
            "application/json",
            use_container_width=True,
        )
    with col2:
        csv_data = export_to_csv(messages)
        st.download_button(
            "CSV",
            csv_data,
            "research_export.csv",
            "text/csv",
            use_container_width=True,
        )
    with col3:
        report = generate_report(messages, sources)
        st.download_button(
            "Report",
            report,
            "research_report.md",
            "text/markdown",
            use_container_width=True,
        )
