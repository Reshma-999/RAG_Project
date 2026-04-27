"""
src/ingest.py
─────────────
Document ingestion pipeline.

Supports: PDF, DOCX, TXT, HTML, Markdown.
Processes 10K+ documents via batched embedding + upsert into ChromaDB.

Usage:
    python -m src.ingest --data-dir ./data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Generator

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from loguru import logger
from tqdm import tqdm

from config import get_settings
from src.vector_store import VectorStoreManager

settings = get_settings()

# ─── Loader registry ──────────────────────────────────────────────────────────

LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
}


def _doc_id(doc: Document) -> str:
    """Deterministic ID based on source path + content hash."""
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
    source = doc.metadata.get("source", "unknown")
    return f"{Path(source).stem}_{content_hash[:10]}"


def load_documents(data_dir: Path) -> list[Document]:
    """
    Recursively load all supported documents from *data_dir*.
    Returns a flat list of raw (un-chunked) Document objects.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_docs: list[Document] = []
    files = list(data_dir.rglob("*"))
    supported = [f for f in files if f.suffix.lower() in LOADER_MAP]

    logger.info(f"Found {len(supported)} supported files in {data_dir}")

    for filepath in tqdm(supported, desc="Loading files"):
        loader_cls = LOADER_MAP[filepath.suffix.lower()]
        try:
            loader = loader_cls(str(filepath))
            docs = loader.load()
            # Inject consistent source metadata
            for doc in docs:
                doc.metadata["source"] = str(filepath)
                doc.metadata["filename"] = filepath.name
                doc.metadata["file_type"] = filepath.suffix.lower()
            all_docs.extend(docs)
        except Exception as exc:
            logger.warning(f"Failed to load {filepath}: {exc}")

    logger.info(f"Loaded {len(all_docs)} raw documents")
    return all_docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    Chunk size and overlap are read from settings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    # Attach stable IDs
    for chunk in chunks:
        chunk.metadata["doc_id"] = _doc_id(chunk)
    logger.info(f"Split into {len(chunks)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    return chunks


def _batched(items: list, size: int) -> Generator[list, None, None]:
    """Yield successive batches of *size* from *items*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ingest(data_dir: Path | None = None) -> int:
    """
    Full ingestion pipeline:
      1. Load raw documents
      2. Split into chunks
      3. Embed + upsert into ChromaDB in batches

    Returns the total number of chunks ingested.
    """
    data_dir = Path(data_dir or settings.data_dir)
    raw_docs = load_documents(data_dir)

    if not raw_docs:
        logger.warning("No documents loaded — nothing to ingest.")
        return 0

    chunks = split_documents(raw_docs)
    vs = VectorStoreManager()

    total_ingested = 0
    batches = list(_batched(chunks, settings.batch_size))
    logger.info(f"Ingesting {len(chunks)} chunks in {len(batches)} batches …")

    for i, batch in enumerate(tqdm(batches, desc="Ingesting batches"), start=1):
        try:
            vs.add_documents(batch)
            total_ingested += len(batch)
        except Exception as exc:
            logger.error(f"Batch {i} failed: {exc}")

    logger.success(f"Ingestion complete: {total_ingested} chunks stored.")
    return total_ingested


def save_ingestion_report(
    n_files: int,
    n_chunks: int,
    output_path: Path = Path("./data/ingestion_report.json"),
) -> None:
    """Write a simple ingestion summary to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"files_processed": n_files, "chunks_stored": n_chunks}
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Report saved to {output_path}")


# ─── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir,
        help="Directory containing raw documents",
    )
    args = parser.parse_args()
    ingest(args.data_dir)
