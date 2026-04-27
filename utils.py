"""
src/utils.py
────────────
Shared utilities used across the project.
"""

from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from loguru import logger


# ─── Timing decorator ─────────────────────────────────────────────────────────

def timeit(label: str = ""):
    """Decorator that logs the execution time of a function."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            tag = label or fn.__name__
            logger.debug(f"[{tag}] elapsed={elapsed:.3f}s")
            return result
        return wrapper
    return decorator


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=indent, ensure_ascii=False))


# ─── Text helpers ─────────────────────────────────────────────────────────────

def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    """Truncate *text* to *max_chars*, appending *suffix* if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def num_tokens(text: str) -> int:
    """Rough token count (1 token ≈ 4 characters for English text)."""
    return len(text) // 4


# ─── Eval dataset generator ───────────────────────────────────────────────────

def generate_sample_eval_dataset(
    output_path: Path = Path("./data/eval_dataset.json"),
    n: int = 20,
) -> list[dict]:
    """
    Generate a synthetic QA eval dataset for smoke-testing.
    In production, replace with real ground-truth QA pairs.
    """
    samples = [
        {
            "question": f"Sample question {i}: What are the key findings in document {i}?",
            "ground_truth": f"The key findings in document {i} relate to the main topic discussed therein.",
        }
        for i in range(1, n + 1)
    ]
    save_json(samples, output_path)
    logger.info(f"Generated {n}-sample eval dataset → {output_path}")
    return samples
