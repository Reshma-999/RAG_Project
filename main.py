"""
main.py
───────
CLI entry point for the RAG pipeline.

Commands:
  ingest   — load and index documents
  query    — ask a single question
  eval     — run the evaluation harness
  serve    — start the FastAPI server
  stats    — print cache statistics

Usage:
  python main.py ingest --data-dir ./data/raw
  python main.py query  "What is the main conclusion of the report?"
  python main.py eval   --dataset ./data/eval_dataset.json --max-samples 50
  python main.py serve
  python main.py stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# ─── Configure logger ─────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
logger.add("logs/rag_pipeline.log", rotation="10 MB", retention="7 days", level="DEBUG")


def cmd_ingest(args: argparse.Namespace) -> None:
    from src.ingest import ingest
    n = ingest(args.data_dir)
    logger.success(f"Ingested {n} chunks.")


def cmd_query(args: argparse.Namespace) -> None:
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(enable_cache=not args.no_cache)
    response = pipeline.query(args.question)
    print(response)


def cmd_eval(args: argparse.Namespace) -> None:
    from src.evaluator import Evaluator
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(enable_cache=False)  # disable cache for honest eval
    ev = Evaluator(pipeline)
    report = ev.run(
        dataset_path=args.dataset,
        max_samples=args.max_samples,
    )
    from config import get_settings
    report.save(get_settings().eval_results_dir)


def cmd_serve(_: argparse.Namespace) -> None:
    import uvicorn
    from api import app
    from config import get_settings

    s = get_settings()
    uvicorn.run(app, host=s.api_host, port=s.api_port)


def cmd_stats(_: argparse.Namespace) -> None:
    from src.redis_cache import SemanticCache
    from src.vector_store import VectorStoreManager

    vs = VectorStoreManager()
    logger.info(f"Vector store: {vs}")

    cache = SemanticCache()
    logger.info(f"Cache: {cache}")


# ─── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="End-to-end RAG Pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index documents into ChromaDB")
    p_ingest.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory with raw documents (default: DATA_DIR from .env)"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", type=str, help="The question to answer")
    p_query.add_argument("--no-cache", action="store_true", help="Bypass semantic cache")
    p_query.set_defaults(func=cmd_query)

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation harness")
    p_eval.add_argument("--dataset", type=Path, default=None,
                        help="Path to QA eval dataset JSON")
    p_eval.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of eval samples")
    p_eval.set_defaults(func=cmd_eval)

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI server")
    p_serve.set_defaults(func=cmd_serve)

    # stats
    p_stats = sub.add_parser("stats", help="Print cache and vector store stats")
    p_stats.set_defaults(func=cmd_stats)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
