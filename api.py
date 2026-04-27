"""
api.py
──────
FastAPI REST API for the RAG pipeline.

Endpoints:
  POST /query          — answer a question
  POST /ingest         — trigger document ingestion
  GET  /health         — health check
  GET  /stats          — cache + vector store statistics
  DELETE /cache        — flush the semantic cache
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag_pipeline import RAGPipeline
from src.redis_cache import SemanticCache
from src.vector_store import VectorStoreManager

app = FastAPI(
    title="RAG Pipeline API",
    description="End-to-end Retrieval-Augmented Generation with Semantic Caching",
    version="1.0.0",
)

# Singleton pipeline (initialised on first request to avoid slow startup)
_pipeline: RAGPipeline | None = None
_cache: SemanticCache | None = None
_vs: VectorStoreManager | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache


def get_vs() -> VectorStoreManager:
    global _vs
    if _vs is None:
        _vs = VectorStoreManager()
    return _vs


# ─── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    use_cache: bool = True
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    latency_s: float
    from_cache: bool


class IngestRequest(BaseModel):
    data_dir: str | None = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    pipeline.enable_cache = req.use_cache

    try:
        resp = pipeline.query(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        question=resp.query,
        answer=resp.answer,
        sources=resp.sources,
        latency_s=resp.latency_s,
        from_cache=resp.from_cache,
    )


@app.post("/ingest")
def ingest_documents(req: IngestRequest) -> dict:
    from pathlib import Path
    from src.ingest import ingest as run_ingest

    try:
        n = run_ingest(Path(req.data_dir) if req.data_dir else None)
        return {"status": "ok", "chunks_ingested": n}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats")
def stats() -> dict:
    vs = get_vs()
    cache = get_cache()
    cache_stats = cache.get_stats()
    return {
        "vector_store": {
            "collection": vs.collection_name,
            "document_count": vs.count(),
        },
        "cache": {
            "total_queries": cache_stats.total_queries,
            "cache_hits": cache_stats.cache_hits,
            "cache_misses": cache_stats.cache_misses,
            "hit_rate": round(cache_stats.hit_rate, 4),
            "api_call_reduction": round(cache_stats.redundant_api_call_reduction, 4),
            "total_latency_saved_s": round(cache_stats.total_latency_saved_s, 2),
        },
    }


@app.delete("/cache")
def flush_cache() -> dict:
    cache = get_cache()
    n = cache.flush()
    return {"status": "ok", "entries_deleted": n}
