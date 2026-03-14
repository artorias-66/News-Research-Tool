"""
Async URL ingestion pipeline with retry logic.

Fetches web content concurrently using aiohttp, applies exponential backoff
on failures, and processes documents into chunks for indexing.
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional

from src.utils import clean_text, validate_url, hash_content, setup_logger, timed
from src.exceptions import URLFetchError, URLValidationError

logger = setup_logger(__name__)

# ───────────────────────────── Constants ─────────────────────────────

MAX_RETRIES = 3
BACKOFF_BASE = 1  # seconds (1s → 2s → 4s)
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ───────────────────────── HTML Extraction ───────────────────────────

def extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML, stripping non-content elements.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned plain text extracted from the HTML body.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)


# ───────────────────── Async Fetch with Retry ────────────────────────

async def _fetch_url_async(
    session: aiohttp.ClientSession,
    url: str,
) -> Document:
    """Fetch a single URL with exponential backoff retry.

    Args:
        session: aiohttp client session.
        url: URL to fetch.

    Returns:
        LangChain Document with page content and source metadata.

    Raises:
        URLFetchError: After all retries are exhausted.
    """
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    text = extract_text_from_html(html)

                    if len(text.strip()) < 50:
                        raise URLFetchError(url, message=f"Extracted content too short ({len(text)} chars).")

                    logger.info(f"Fetched {url} ({len(text)} chars)")
                    return Document(page_content=text, metadata={"source": url})
                else:
                    raise URLFetchError(url, status_code=response.status)

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {url}: {e}. "
                f"Retrying in {wait}s..."
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait)

    raise URLFetchError(url, message=f"All {MAX_RETRIES} retries exhausted for {url}.")


async def fetch_urls_async(urls: list[str]) -> list[Document]:
    """Fetch multiple URLs concurrently.

    Uses aiohttp to fetch all URLs in parallel, with per-URL retry logic.
    Failed URLs are logged but do not block successful ones.

    Args:
        urls: List of validated URL strings.

    Returns:
        List of successfully fetched Documents.
    """
    documents: list[Document] = []

    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_url_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch {url}: {result}")
        else:
            documents.append(result)

    if not documents:
        raise URLFetchError(
            url=", ".join(urls),
            message="All URLs failed to fetch. No content to process.",
        )

    return documents


# ────────────────────── Synchronous Fallback ─────────────────────────

def _fetch_url_sync(url: str) -> Document:
    """Synchronous URL fetch with retry — fallback for environments without async support.

    Args:
        url: URL to fetch.

    Returns:
        LangChain Document with extracted content.

    Raises:
        URLFetchError: After all retries are exhausted.
    """
    import time

    headers = {"User-Agent": USER_AGENT}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                text = extract_text_from_html(response.text)
                if len(text.strip()) < 50:
                    raise URLFetchError(url, message=f"Extracted content too short ({len(text)} chars).")
                logger.info(f"Fetched {url} ({len(text)} chars)")
                return Document(page_content=text, metadata={"source": url})
            else:
                raise URLFetchError(url, status_code=response.status_code)
        except requests.RequestException as e:
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {url}: {e}. Retrying in {wait}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    raise URLFetchError(url, message=f"All {MAX_RETRIES} retries exhausted for {url}.")


# ───────────────────────── Public API ────────────────────────────────

@timed
def load_urls(urls: list[str]) -> list[Document]:
    """Load content from a list of URLs.

    Validates URLs, deduplicates, then fetches concurrently using async I/O
    with a synchronous fallback for restricted environments.

    Args:
        urls: List of URL strings to fetch.

    Returns:
        List of LangChain Documents with page content and source metadata.

    Raises:
        URLValidationError: If any URL is invalid.
        URLFetchError: If all URLs fail to fetch.
    """
    # Validate URLs
    validated: list[str] = []
    for url in urls:
        validated.append(validate_url(url))

    # Deduplicate URLs
    seen_hashes: set[str] = set()
    unique_urls: list[str] = []
    for url in validated:
        url_hash = hash_content(url.lower().rstrip("/"))
        if url_hash not in seen_hashes:
            seen_hashes.add(url_hash)
            unique_urls.append(url)
        else:
            logger.info(f"Skipping duplicate URL: {url}")

    logger.info(f"Loading {len(unique_urls)} unique URLs (from {len(urls)} provided)...")

    # Try async first, fall back to sync
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g., Streamlit) — use sync fallback
            documents = []
            for url in unique_urls:
                try:
                    documents.append(_fetch_url_sync(url))
                except URLFetchError as e:
                    logger.error(str(e))
            if not documents:
                raise URLFetchError(
                    url=", ".join(unique_urls),
                    message="All URLs failed to fetch.",
                )
            return documents
        else:
            return asyncio.run(fetch_urls_async(unique_urls))
    except Exception as e:
        if isinstance(e, (URLFetchError, URLValidationError)):
            raise
        logger.warning(f"Async fetch failed: {e}. Falling back to sync...")
        documents = []
        for url in unique_urls:
            try:
                documents.append(_fetch_url_sync(url))
            except URLFetchError as err:
                logger.error(str(err))
        if not documents:
            raise URLFetchError(
                url=", ".join(unique_urls),
                message="All URLs failed to fetch.",
            )
        return documents


@timed
def process_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Clean and split documents into chunks for indexing.

    Args:
        documents: List of LangChain Documents to process.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Character overlap between adjacent chunks.

    Returns:
        List of chunked Documents ready for embedding.
    """
    logger.info(f"Processing {len(documents)} documents...")

    # Clean text
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    # Filter out very small chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
