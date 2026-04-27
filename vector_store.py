"""
src/vector_store.py
───────────────────
ChromaDB vector store wrapper.

Provides a thin, reusable interface around Chroma so the rest of the
codebase never imports chromadb directly.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from config import get_settings

settings = get_settings()


class VectorStoreManager:
    """
    Manages the ChromaDB collection used for document retrieval.

    Attributes
    ----------
    collection_name : str
        Name of the Chroma collection.
    persist_dir : Path
        Directory where Chroma persists data to disk.
    embeddings : OpenAIEmbeddings
        Embedding function used for both ingestion and retrieval.
    vectorstore : Chroma
        The underlying LangChain Chroma instance.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: Path | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.vectorstore = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        logger.debug(
            f"VectorStoreManager initialised — collection='{self.collection_name}', "
            f"persist_dir='{self.persist_dir}'"
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and upsert *docs* into the vector store."""
        if not docs:
            return
        ids = [doc.metadata.get("doc_id", f"doc_{i}") for i, doc in enumerate(docs)]
        self.vectorstore.add_documents(documents=docs, ids=ids)
        logger.debug(f"Upserted {len(docs)} documents.")

    def delete_collection(self) -> None:
        """Drop and recreate the collection (nuclear option for re-indexing)."""
        self._client.delete_collection(self.collection_name)
        self.vectorstore = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        logger.warning(f"Collection '{self.collection_name}' deleted and recreated.")

    # ── Read ───────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """
        Return the top-*k* most relevant documents for *query*.

        Parameters
        ----------
        query : str
            The user query string.
        k : int, optional
            Number of results to return (defaults to settings.top_k).
        score_threshold : float, optional
            Minimum cosine similarity score (defaults to settings.retrieval_score_threshold).
        """
        k = k or settings.top_k
        threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold

        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        filtered = [
            doc
            for doc, score in results
            if score >= threshold
        ]
        for doc, score in results:
            doc.metadata["retrieval_score"] = round(score, 4)

        logger.debug(
            f"Retrieved {len(filtered)}/{k} docs above threshold={threshold} for query='{query[:60]}…'"
        )
        return filtered

    def as_retriever(self, **kwargs):
        """Return a LangChain-compatible retriever for use in chains."""
        return self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": settings.top_k,
                "score_threshold": settings.retrieval_score_threshold,
                **kwargs,
            },
        )

    # ── Stats ──────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of documents currently stored."""
        collection = self._client.get_collection(self.collection_name)
        return collection.count()

    def __repr__(self) -> str:
        return (
            f"VectorStoreManager(collection='{self.collection_name}', "
            f"docs={self.count()})"
        )
