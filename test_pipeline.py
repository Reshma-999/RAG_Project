"""
tests/test_pipeline.py
───────────────────────
Tests for the RAG pipeline components.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings(monkeypatch):
    """Patch settings to avoid needing real env vars during tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")


@pytest.fixture
def sample_documents():
    from langchain_core.documents import Document
    return [
        Document(
            page_content="The quarterly revenue increased by 15% year-over-year.",
            metadata={"source": "report.pdf", "filename": "report.pdf", "doc_id": "doc_001"},
        ),
        Document(
            page_content="Customer satisfaction scores reached 92% in Q3.",
            metadata={"source": "survey.pdf", "filename": "survey.pdf", "doc_id": "doc_002"},
        ),
        Document(
            page_content="New product line launched in APAC markets during October.",
            metadata={"source": "news.txt", "filename": "news.txt", "doc_id": "doc_003"},
        ),
    ]


# ─── Ingestion tests ──────────────────────────────────────────────────────────

class TestSplitDocuments:
    def test_split_produces_chunks(self, sample_documents):
        from src.ingest import split_documents
        with patch("src.ingest.settings") as mock_s:
            mock_s.chunk_size = 100
            mock_s.chunk_overlap = 10
            chunks = split_documents(sample_documents)
            assert len(chunks) >= len(sample_documents)

    def test_chunks_have_doc_id(self, sample_documents):
        from src.ingest import split_documents
        with patch("src.ingest.settings") as mock_s:
            mock_s.chunk_size = 100
            mock_s.chunk_overlap = 10
            chunks = split_documents(sample_documents)
            for chunk in chunks:
                assert "doc_id" in chunk.metadata


# ─── Retriever tests ──────────────────────────────────────────────────────────

class TestRetriever:
    def test_jaccard_similarity_identical(self):
        from src.retriever import Retriever
        r = Retriever.__new__(Retriever)
        assert r._jaccard_similarity("hello world", "hello world") == 1.0

    def test_jaccard_similarity_disjoint(self):
        from src.retriever import Retriever
        r = Retriever.__new__(Retriever)
        assert r._jaccard_similarity("foo bar", "baz qux") == 0.0

    def test_build_citation(self, sample_documents):
        from src.retriever import Retriever
        doc = sample_documents[0]
        doc.metadata["retrieval_score"] = 0.87
        citation = Retriever._build_citation(doc, rank=1)
        assert citation["rank"] == 1
        assert citation["filename"] == "report.pdf"
        assert citation["score"] == 0.87
        assert len(citation["snippet"]) <= 200

    def test_format_context(self, sample_documents):
        from src.retriever import Retriever, RetrievalResult
        r = Retriever.__new__(Retriever)
        results = [
            RetrievalResult(doc=doc, score=0.9 - i * 0.1, rank=i + 1)
            for i, doc in enumerate(sample_documents)
        ]
        context = r.format_context(results)
        assert "[1]" in context
        assert "[2]" in context


# ─── Cache tests ──────────────────────────────────────────────────────────────

class TestCacheStats:
    def test_hit_rate_zero_queries(self):
        from src.redis_cache import CacheStats
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        from src.redis_cache import CacheStats
        stats = CacheStats(total_queries=10, cache_hits=4, cache_misses=6)
        assert stats.hit_rate == pytest.approx(0.4)

    def test_api_reduction_equals_hit_rate(self):
        from src.redis_cache import CacheStats
        stats = CacheStats(total_queries=100, cache_hits=38, cache_misses=62)
        assert stats.redundant_api_call_reduction == pytest.approx(0.38)


# ─── Citation accuracy tests ─────────────────────────────────────────────────

class TestCitationAccuracy:
    def test_valid_citations(self):
        from src.evaluator import Evaluator
        sources = [{"filename": "a.pdf"}, {"filename": "b.pdf"}]
        answer = "According to [1], revenue grew. Also [2] confirms this."
        acc = Evaluator._citation_accuracy(answer, sources)
        assert acc == 1.0

    def test_invalid_citation(self):
        from src.evaluator import Evaluator
        sources = [{"filename": "a.pdf"}]
        answer = "According to [1] and [5], revenue grew."  # [5] is invalid
        acc = Evaluator._citation_accuracy(answer, sources)
        assert acc == pytest.approx(0.5)

    def test_no_citations(self):
        from src.evaluator import Evaluator
        sources = [{"filename": "a.pdf"}]
        answer = "Revenue grew significantly this quarter."
        acc = Evaluator._citation_accuracy(answer, sources)
        assert acc == 1.0


# ─── Pipeline integration smoke test ─────────────────────────────────────────

class TestRAGPipeline:
    @patch("src.rag_pipeline.OpenAI")
    @patch("src.rag_pipeline.Retriever")
    @patch("src.rag_pipeline.SemanticCache")
    def test_query_on_cache_miss(self, MockCache, MockRetriever, MockOpenAI):
        from langchain_core.documents import Document
        from src.rag_pipeline import RAGPipeline
        from src.retriever import RetrievalResult

        # Cache returns None (miss)
        mock_cache = MockCache.return_value
        mock_cache.get.return_value = None

        # Retriever returns one result
        doc = Document(
            page_content="Revenue grew by 15%.",
            metadata={"filename": "report.pdf", "retrieval_score": 0.9},
        )
        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = [
            RetrievalResult(doc=doc, score=0.9, rank=1, citation={"rank": 1, "filename": "report.pdf", "source": "report.pdf", "score": 0.9, "snippet": "Revenue grew by 15%."})
        ]
        mock_retriever.format_context.return_value = "[1] report.pdf\nRevenue grew by 15%."

        # LLM returns a canned answer
        mock_llm = MockOpenAI.return_value
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "Revenue grew by 15% [1]."
        )

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            cache=mock_cache,
            enable_cache=True,
        )
        # Inject the mock OpenAI client directly
        pipeline._client = mock_llm

        response = pipeline.query("What was the revenue growth?")

        assert "15%" in response.answer
        assert response.from_cache is False
        mock_cache.set.assert_called_once()

    @patch("src.rag_pipeline.OpenAI")
    @patch("src.rag_pipeline.Retriever")
    @patch("src.rag_pipeline.SemanticCache")
    def test_query_on_cache_hit(self, MockCache, MockRetriever, MockOpenAI):
        from src.redis_cache import CacheEntry
        from src.rag_pipeline import RAGPipeline

        mock_cache = MockCache.return_value
        mock_cache.get.return_value = CacheEntry(
            query="What was the revenue growth?",
            answer="Revenue grew by 15% [1].",
            sources=[{"filename": "report.pdf"}],
            query_vector=[0.1] * 1536,
        )

        pipeline = RAGPipeline(
            retriever=MockRetriever.return_value,
            cache=mock_cache,
            enable_cache=True,
        )

        response = pipeline.query("What was the revenue growth?")
        assert response.from_cache is True
        MockRetriever.return_value.retrieve.assert_not_called()


# ─── Utility tests ────────────────────────────────────────────────────────────

class TestUtils:
    def test_truncate_short(self):
        from src.utils import truncate
        assert truncate("hello", 100) == "hello"

    def test_truncate_long(self):
        from src.utils import truncate
        result = truncate("a" * 300, 200)
        assert len(result) <= 200
        assert result.endswith("…")

    def test_num_tokens(self):
        from src.utils import num_tokens
        assert num_tokens("hello world") == 2  # 11 chars // 4
