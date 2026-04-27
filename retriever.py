"""
src/retriever.py
────────────────
Retrieval layer.

Wraps the vector store with:
  • Maximal Marginal Relevance (MMR) for diversity
  • Optional cross-encoder reranking for precision
  • Citation metadata attachment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from config import get_settings
from src.vector_store import VectorStoreManager

settings = get_settings()


@dataclass
class RetrievalResult:
    """Container for a single retrieval result."""
    doc: Document
    score: float
    rank: int
    citation: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        source = self.doc.metadata.get("filename", "Unknown source")
        return f"[{self.rank}] ({source})\n{self.doc.page_content.strip()}"


class Retriever:
    """
    High-level retriever that adds MMR diversity and citation tracking.

    Parameters
    ----------
    vs_manager : VectorStoreManager, optional
        Pre-constructed manager; created fresh if not provided.
    use_mmr : bool
        If True, use Maximal Marginal Relevance instead of pure similarity.
    mmr_lambda : float
        MMR trade-off between relevance and diversity (1.0 = pure relevance).
    """

    def __init__(
        self,
        vs_manager: VectorStoreManager | None = None,
        use_mmr: bool = True,
        mmr_lambda: float = 0.6,
    ) -> None:
        self.vs = vs_manager or VectorStoreManager()
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    # ── Core retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top-*k* relevant documents for *query*.

        Returns a ranked list of RetrievalResult objects.
        """
        k = k or settings.top_k
        threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

        if self.use_mmr:
            docs = self._mmr_retrieve(query, k, threshold)
        else:
            docs = self.vs.similarity_search(query, k=k, score_threshold=threshold)

        results = []
        for rank, doc in enumerate(docs, start=1):
            score = doc.metadata.get("retrieval_score", 0.0)
            citation = self._build_citation(doc, rank)
            doc.metadata["citation"] = citation
            results.append(RetrievalResult(doc=doc, score=score, rank=rank, citation=citation))

        logger.debug(f"Retrieved {len(results)} documents for query='{query[:60]}…'")
        return results

    def _mmr_retrieve(
        self, query: str, k: int, threshold: float
    ) -> list[Document]:
        """
        Maximal Marginal Relevance retrieval.

        Fetches 3× the desired k for the candidate pool, then re-ranks
        via cosine similarity vs diversity to reduce redundancy.
        """
        candidate_pool = self.vs.similarity_search(
            query, k=k * 3, score_threshold=threshold
        )
        if not candidate_pool:
            return []

        # Greedy MMR selection
        selected: list[Document] = []
        remaining = list(candidate_pool)

        while len(selected) < k and remaining:
            best_doc = None
            best_score = -float("inf")

            for doc in remaining:
                relevance = doc.metadata.get("retrieval_score", 0.0)

                # Diversity penalty: max similarity to already-selected docs
                if selected:
                    max_sim = max(
                        self._jaccard_similarity(doc.page_content, s.page_content)
                        for s in selected
                    )
                    mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                else:
                    mmr_score = relevance

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc

            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)

        return selected

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Lightweight token-level Jaccard similarity for MMR diversity calc."""
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    # ── Citation building ──────────────────────────────────────────────────────

    @staticmethod
    def _build_citation(doc: Document, rank: int) -> dict[str, Any]:
        meta = doc.metadata
        return {
            "rank": rank,
            "source": meta.get("source", "Unknown"),
            "filename": meta.get("filename", "Unknown"),
            "page": meta.get("page", None),
            "score": meta.get("retrieval_score", 0.0),
            "snippet": doc.page_content[:200].strip(),
        }

    # ── Context formatting ────────────────────────────────────────────────────

    def format_context(self, results: list[RetrievalResult]) -> str:
        """
        Format retrieval results into a numbered context block
        suitable for injection into a prompt.
        """
        return "\n\n---\n\n".join(r.to_context_string() for r in results)
