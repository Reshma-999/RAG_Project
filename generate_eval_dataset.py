"""
scripts/generate_eval_dataset.py
──────────────────────────────────
Generate a synthetic QA evaluation dataset from your indexed documents.

Uses the LLM to produce question/answer pairs from sampled chunks.
Run AFTER ingestion.

Usage:
    python scripts/generate_eval_dataset.py --n 100 --output ./data/eval_dataset.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from config import get_settings
from src.vector_store import VectorStoreManager

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

QA_GENERATION_PROMPT = """\
Given the following document excerpt, generate ONE factual question and its answer.
The question should be specific and answerable from the text alone.
Respond in JSON format: {{"question": "...", "ground_truth": "..."}}

Document:
{chunk}
"""


def generate_qa_pair(chunk_text: str) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "user", "content": QA_GENERATION_PROMPT.format(chunk=chunk_text[:800])},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning(f"Failed to generate QA pair: {exc}")
        return None


def main(n: int, output: Path) -> None:
    vs = VectorStoreManager()
    total = vs.count()
    if total == 0:
        logger.error("Vector store is empty. Run ingestion first.")
        return

    # Sample random chunks from the store via broad queries
    sample_queries = [
        "main findings", "key results", "introduction", "conclusion",
        "methodology", "data analysis", "summary", "overview",
    ]
    seen_content: set[str] = set()
    sampled_chunks: list[str] = []

    for q in sample_queries * (n // len(sample_queries) + 1):
        if len(sampled_chunks) >= n * 2:
            break
        docs = vs.similarity_search(q, k=10, score_threshold=0.0)
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen_content and len(doc.page_content) > 100:
                seen_content.add(key)
                sampled_chunks.append(doc.page_content)

    random.shuffle(sampled_chunks)
    sampled_chunks = sampled_chunks[:n]

    logger.info(f"Generating {len(sampled_chunks)} QA pairs …")
    dataset = []
    for chunk in tqdm(sampled_chunks):
        pair = generate_qa_pair(chunk)
        if pair and "question" in pair and "ground_truth" in pair:
            dataset.append(pair)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    logger.success(f"Saved {len(dataset)} QA pairs → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of QA pairs to generate")
    parser.add_argument("--output", type=Path, default=Path("./data/eval_dataset.json"))
    args = parser.parse_args()
    main(args.n, args.output)
