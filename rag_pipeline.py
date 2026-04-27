"""
src/rag_pipeline.py
───────────────────
End-to-end RAG pipeline.

Flow
────
  User Query
      │
      ▼
  SemanticCache.get()  ──HIT──►  Return cached answer   (< 1 s)
      │
      MISS
      │
      ▼
  Retriever.retrieve()           (vector similarity + MMR)
      │
      ▼
  Prompt assembly                (system + context + query)
      │
      ▼
  OpenAI ChatCompletion          (~3 s on miss)
      │
      ▼
  SemanticCache.set()            (store for future hits)
      │
      ▼
  Return RAGResponse
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from src.redis_cache import SemanticCache
from src.retriever import Retriever, RetrievalResult

settings = get_settings()

# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, helpful research assistant.
Answer the user's question using ONLY the provided context documents.
Rules:
- Be factual, concise, and well-structured.
- Cite sources by their [number] inline (e.g. "According to [1]…").
- If the context does not contain enough information, say so clearly — do not invent facts.
- Where possible, include specific figures, dates, or names from the context.
"""

QUERY_TEMPLATE = """\
Context Documents:
{context}

---
Question: {query}

Answer (with inline citations):"""


# ─── Response container ───────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[dict[str, Any]]
    latency_s: float
    from_cache: bool
    retrieval_scores: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        cache_tag = "CACHE HIT" if self.from_cache else "LLM"
        return (
            f"[{cache_tag} | {self.latency_s:.2f}s]\n"
            f"Q: {self.query}\n\n"
            f"A: {self.answer}\n\n"
            f"Sources: {[s['filename'] for s in self.sources]}"
        )


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates retrieval, caching, and generation.

    Parameters
    ----------
    retriever : Retriever, optional
    cache : SemanticCache, optional
    enable_cache : bool
        Set to False to bypass the semantic cache (useful for eval).
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        cache: SemanticCache | None = None,
        enable_cache: bool = True,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.cache = cache or SemanticCache()
        self.enable_cache = enable_cache
        self._client = OpenAI(api_key=settings.openai_api_key)
        logger.info(
            f"RAGPipeline initialised (cache={'enabled' if enable_cache else 'disabled'})"
        )

    # ── Public ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """
        Answer *question* using the full RAG pipeline.

        1. Check semantic cache.
        2. If miss: retrieve → generate → cache.
        3. Return RAGResponse.
        """
        t0 = time.perf_counter()

        # ── Cache look-up ────────────────────────────────────────────────────
        if self.enable_cache:
            cached = self.cache.get(question)
            if cached:
                latency = time.perf_counter() - t0
                return RAGResponse(
                    query=question,
                    answer=cached.answer,
                    sources=cached.sources,
                    latency_s=latency,
                    from_cache=True,
                )

        # ── Retrieval ────────────────────────────────────────────────────────
        results: list[RetrievalResult] = self.retriever.retrieve(question)
        if not results:
            answer = (
                "I could not find relevant information in the knowledge base "
                "to answer your question."
            )
            return RAGResponse(
                query=question,
                answer=answer,
                sources=[],
                latency_s=time.perf_counter() - t0,
                from_cache=False,
            )

        context = self.retriever.format_context(results)
        sources = [r.citation for r in results]

        # ── Generation ───────────────────────────────────────────────────────
        answer = self._generate(question, context)

        latency = time.perf_counter() - t0
        logger.info(f"LLM response in {latency:.2f}s for query='{question[:60]}…'")

        # ── Store in cache ───────────────────────────────────────────────────
        if self.enable_cache:
            self.cache.set(question, answer, sources)

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            latency_s=latency,
            from_cache=False,
            retrieval_scores=[r.score for r in results],
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _generate(self, query: str, context: str) -> str:
        """Call the OpenAI Chat API with retry logic."""
        prompt = QUERY_TEMPLATE.format(context=context, query=query)
        response = self._client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,    # low temp for factual faithfulness
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    # ── Batch ──────────────────────────────────────────────────────────────────

    def batch_query(self, questions: list[str]) -> list[RAGResponse]:
        """Run a list of questions through the pipeline sequentially."""
        responses = []
        for q in questions:
            try:
                responses.append(self.query(q))
            except Exception as exc:
                logger.error(f"Failed to process query '{q[:60]}…': {exc}")
                responses.append(
                    RAGResponse(
                        query=q,
                        answer=f"Error: {exc}",
                        sources=[],
                        latency_s=0.0,
                        from_cache=False,
                    )
                )
        return responses
