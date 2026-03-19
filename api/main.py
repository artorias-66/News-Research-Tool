"""
FastAPI application with RESTful API for the RAG pipeline.

Provides endpoints for URL ingestion, querying, exporting,
health checks, and metrics. Decouples the RAG engine from the
Streamlit frontend for independent testability and scalability.
"""


import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ingest import load_urls, process_documents
from src.vector_store import create_vector_store, load_vector_store, load_index_metadata
from src.retriever import BM25Retriever
from src.rag import get_answer, get_cache, get_memory
from src.export import export_to_json, export_to_csv, generate_report
from src.exceptions import ResearchToolError
from api.middleware import TokenBucketRateLimiter
from src.utils import setup_logger

logger = setup_logger(__name__)

# ─────────────────── Pydantic Models ─────────────────────────────────


class IngestRequest(BaseModel):
    """Request body for URL ingestion."""
    urls: list[str] = Field(..., min_length=1, description="List of URLs to process")
    chunk_size: int = Field(1000, ge=100, le=5000, description="Chunk size in characters")
    chunk_overlap: int = Field(200, ge=0, le=500, description="Overlap between chunks")


class IngestResponse(BaseModel):
    """Response after successful ingestion."""
    message: str
    chunks_created: int
    urls_processed: int
    processing_time_ms: float


class QueryRequest(BaseModel):
    """Request body for RAG query."""
    question: str = Field(..., min_length=1, description="Research question")


class QueryResponse(BaseModel):
    """Response with RAG answer."""
    answer: str
    sources: list[str]
    retrieval_method: str
    is_relevant: bool
    has_llm: bool
    from_cache: bool


class ExportRequest(BaseModel):
    """Request body for export."""
    format: str = Field("json", description="Export format: 'json', 'csv', or 'report'")
    messages: list[dict[str, str]] = Field(..., description="Chat messages to export")
    sources: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_store_loaded: bool
    bm25_loaded: bool
    index_metadata: Optional[dict] = None


class MetricsResponse(BaseModel):
    """Metrics response."""
    cache: dict
    memory_turns: int
    index_metadata: Optional[dict] = None


# ────────────────────── App Lifecycle ────────────────────────────────

# Module-level state
_state: dict = {
    "vector_store": None,
    "bm25_retriever": BM25Retriever(),
    "rate_limiter": TokenBucketRateLimiter(max_tokens=20, refill_rate=2.0),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load vector store and BM25 index on startup."""
    logger.info("Starting up — loading indices...")
    try:
        _state["vector_store"] = load_vector_store()
        _state["bm25_retriever"].load()
    except Exception as e:
        logger.warning(f"Index loading failed (will require re-ingestion): {e}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Equity Research Tool API",
    description="Hybrid RAG pipeline with BM25 + FAISS + Cross-Encoder re-ranking",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────── Rate Limit Dependency ────────────────────────────


def _check_rate_limit(request: Request) -> None:
    """Check rate limit and raise HTTPException if exceeded."""
    client_ip = request.client.host if request.client else "unknown"
    limiter = _state["rate_limiter"]
    if not limiter.consume(client_ip):
        retry_after = limiter.get_retry_after(client_ip)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "RATE_LIMIT_ERROR",
                "message": "Too many requests",
                "retry_after": round(retry_after, 1),
            },
        )


# ────────────────────── Exception Handler ────────────────────────────


@app.exception_handler(ResearchToolError)
async def research_tool_error_handler(request: Request, exc: ResearchToolError):
    """Map custom exceptions to proper HTTP responses."""
    status_map = {
        "URL_FETCH_ERROR": 502,
        "URL_VALIDATION_ERROR": 400,
        "EMBEDDING_ERROR": 500,
        "VECTOR_STORE_ERROR": 500,
        "LLM_ERROR": 502,
        "RATE_LIMIT_ERROR": 429,
        "EXPORT_ERROR": 500,
    }
    status_code = status_map.get(exc.error_code, 500)
    raise HTTPException(status_code=status_code, detail=exc.to_dict())


# ──────────────────────── Endpoints ──────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        vector_store_loaded=_state["vector_store"] is not None,
        bm25_loaded=_state["bm25_retriever"].bm25 is not None,
        index_metadata=load_index_metadata(),
    )


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return cache and system metrics."""
    return MetricsResponse(
        cache=get_cache().metrics.to_dict(),
        memory_turns=get_memory().turn_count,
        index_metadata=load_index_metadata(),
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_urls(request: Request, body: IngestRequest):
    """Process URLs: fetch, chunk, embed, and index."""
    _check_rate_limit(request)

    start = time.perf_counter()

    documents = load_urls(body.urls)
    chunks = process_documents(documents, body.chunk_size, body.chunk_overlap)

    # Build FAISS index
    vector_store = create_vector_store(chunks)
    _state["vector_store"] = vector_store

    # Build BM25 index
    _state["bm25_retriever"].build_index(chunks)
    _state["bm25_retriever"].save()

    elapsed = (time.perf_counter() - start) * 1000

    return IngestResponse(
        message="Processing complete",
        chunks_created=len(chunks),
        urls_processed=len(documents),
        processing_time_ms=round(elapsed, 1),
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """Query the RAG pipeline."""
    _check_rate_limit(request)

    if _state["vector_store"] is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Call /api/ingest first.")

    result = get_answer(
        query=body.question,
        vector_store=_state["vector_store"],
        bm25_retriever=_state["bm25_retriever"],
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieval_method=result.get("retrieval_method", "unknown"),
        is_relevant=result.get("is_relevant", False),
        has_llm=result.get("has_llm", False),
        from_cache=result.get("from_cache", False),
    )


@app.post("/api/export")
async def export_data(body: ExportRequest):
    """Export research data in the specified format."""
    metadata = load_index_metadata()

    if body.format == "json":
        return {"content": export_to_json(body.messages, metadata), "format": "json"}
    elif body.format == "csv":
        return {"content": export_to_csv(body.messages), "format": "csv"}
    elif body.format == "report":
        return {
            "content": generate_report(body.messages, body.sources, metadata),
            "format": "markdown",
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {body.format}")
