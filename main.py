import os
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool 📰")
st.sidebar.title("News Article Links")
st.sidebar.info("✅ Using FREE local embeddings - No API needed!")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "embeddings_store.pkl"

main_placeholder = st.empty()


def clean_text(text):
    """Clean unwanted text patterns"""
    import re

    # Remove common noise patterns - exact matches or standalone words
    noise_patterns = [
        r'\bTags\b', r'\bShare\b', r'\bTweet\b', r'\bFollow\b', r'\bSubscribe\b',
        r'\bAdvertisement\b', r'\bCookie Policy\b', r'\bPrivacy Policy\b',
        r'\bRead More\b', r'\bClick Here\b', r'\bComments\b', r'\bRelated Articles\b',
        r'\bSign up\b', r'\bLog in\b', r'\bNewsletter\b'
    ]

    # Remove noise patterns
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove multiple spaces and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    # Remove very short lines (likely navigation/UI elements)
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Keep lines that are either long enough or part of a sentence
        if line and (len(line) > 30 or line.endswith('.') or line.endswith('?')):
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def split_text(text, chunk_size=1000, overlap=200):
    """Simple text splitter"""
    text = clean_text(text)  # Clean first
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


if process_url_clicked:
    if not urls:
        st.error("Please enter at least one URL")
    else:
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("📥 Loading data from URLs...")
            data = loader.load()

            # Split into chunks
            main_placeholder.text("✂️ Splitting text into chunks...")
            all_chunks = []
            chunk_metadata = []

            for doc in data:
                chunks = split_text(doc.page_content, chunk_size=1000, overlap=200)
                for chunk in chunks:
                    if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                        all_chunks.append(chunk)
                        chunk_metadata.append({
                            'source': doc.metadata.get('source', 'Unknown'),
                            'text': chunk
                        })

            st.info(f"📊 Created {len(all_chunks)} text chunks from {len(urls)} URL(s)")

            # Load embedding model
            main_placeholder.text("🧠 Loading embedding model (first time takes 1-2 min)...")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            # Create embeddings
            main_placeholder.text("🔢 Creating embeddings...")
            embeddings = model.encode(all_chunks, show_progress_bar=False)

            # Save embeddings and metadata
            with open(file_path, "wb") as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'chunks': all_chunks,
                    'metadata': chunk_metadata,
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
                }, f)

            main_placeholder.success("✅ Processing Complete! You can now ask questions.")
            st.balloons()

        except Exception as e:
            main_placeholder.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Query section
st.markdown("---")
query = st.text_input("💬 Ask a question about the articles:")

if query:
    if os.path.exists(file_path):
        try:
            # Load stored data
            with open(file_path, "rb") as f:
                stored_data = pickle.load(f)

            embeddings = stored_data['embeddings']
            chunks = stored_data['chunks']
            metadata = stored_data['metadata']

            # Load model and encode query
            with st.spinner("🔍 Searching for relevant information..."):
                model = SentenceTransformer(stored_data['model_name'])
                query_embedding = model.encode([query])

                # Calculate similarities
                similarities = cosine_similarity(query_embedding, embeddings)[0]

                # Get top 3 most relevant chunks
                top_indices = np.argsort(similarities)[-3:][::-1]

            # Display combined answer
            st.header("💡 Answer")

            # Combine all relevant chunks and remove duplicates
            seen_content = set()
            unique_chunks = []

            for idx in top_indices:
                chunk = chunks[idx].strip()
                # Create a signature to detect duplicates
                chunk_signature = chunk[:100]  # First 100 chars as signature

                if chunk_signature not in seen_content:
                    seen_content.add(chunk_signature)
                    unique_chunks.append(chunk)

            combined_context = "\n\n".join(unique_chunks)

            # Display synthesized answer
            st.write("**Based on the retrieved information:**")
            st.write(combined_context[:2000] + "..." if len(combined_context) > 2000 else combined_context)

            st.markdown("---")

            # Show individual sources for reference
            with st.expander("🔍 View Individual Source Sections"):
                for rank, idx in enumerate(top_indices, 1):
                    similarity_score = similarities[idx]
                    chunk_text = chunks[idx]
                    source = metadata[idx]['source']

                    st.markdown(f"**Section {rank}** (Relevance: {similarity_score:.2%})")
                    st.markdown(f"*Source: {source}*")
                    st.write(chunk_text)
                    st.markdown("---")

            # Display all sources used
            st.subheader("📚 Sources Used:")
            unique_sources = list(set([metadata[idx]['source'] for idx in top_indices]))
            for i, source in enumerate(unique_sources, 1):
                st.write(f"{i}. {source}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    else:
        st.warning("⚠️ Please process URLs first by entering them in the sidebar and clicking 'Process URLs'!")

# Footer
st.markdown("---")
st.caption("💡 Made by Anubhav Verma")