"""
src/evaluator.py
────────────────
Evaluation harness.

Metrics measured
────────────────
  1. Answer Faithfulness        — does the answer stay grounded in context?
  2. Citation Accuracy          — are cited sources actually relevant?
  3. Retrieval Precision        — do retrieved docs contain the answer?
  4. Answer Relevancy           — is the answer relevant to the question?
  5. Context Recall             — is all needed info present in retrieved docs?

Powered by RAGAS + a lightweight custom citation checker.

Usage
─────
    from src.evaluator import Evaluator
    ev = Evaluator(pipeline)
    report = ev.run("./data/eval_dataset.json")
    report.save("./data/eval_results/")
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from config import get_settings
from src.rag_pipeline import RAGPipeline, RAGResponse

settings = get_settings()


# ─── Result containers ────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    sources: list[dict[str, Any]]
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    citation_accuracy: float = 0.0
    latency_s: float = 0.0
    from_cache: bool = False


@dataclass
class EvalReport:
    samples: list[EvalSample] = field(default_factory=list)
    timestamp: str = ""
    summary: dict[str, float] = field(default_factory=dict)

    def compute_summary(self) -> dict[str, float]:
        if not self.samples:
            return {}
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "citation_accuracy",
            "latency_s",
        ]
        summary = {}
        for m in metrics:
            values = [getattr(s, m) for s in self.samples]
            summary[f"mean_{m}"] = round(sum(values) / len(values), 4)
        summary["n_samples"] = len(self.samples)
        summary["cache_hit_rate"] = round(
            sum(1 for s in self.samples if s.from_cache) / len(self.samples), 4
        )
        self.summary = summary
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self.samples])

    def save(self, output_dir: Path | str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full sample detail
        df = self.to_dataframe()
        csv_path = output_dir / f"eval_results_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # Summary JSON
        summary_path = output_dir / f"eval_summary_{self.timestamp}.json"
        summary_path.write_text(json.dumps(self.summary, indent=2))

        logger.success(
            f"Eval report saved → {csv_path} | summary → {summary_path}"
        )

    def print_summary(self) -> None:
        if not self.summary:
            self.compute_summary()
        print("\n" + "═" * 55)
        print("  RAG EVALUATION SUMMARY")
        print("═" * 55)
        for k, v in self.summary.items():
            label = k.replace("mean_", "").replace("_", " ").title()
            value = f"{v:.1%}" if isinstance(v, float) and v <= 1.0 else str(v)
            print(f"  {label:<35} {value}")
        print("═" * 55 + "\n")


# ─── Evaluator ────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs the evaluation harness against a dataset of QA pairs.

    Dataset format (JSON list):
    [
      {
        "question": "What is X?",
        "ground_truth": "X is …"
      },
      ...
    ]
    """

    def __init__(self, pipeline: RAGPipeline | None = None) -> None:
        self.pipeline = pipeline or RAGPipeline()

    # ── Dataset loading ────────────────────────────────────────────────────────

    @staticmethod
    def load_dataset(path: Path | str) -> list[dict[str, str]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Eval dataset not found: {path}")
        data = json.loads(path.read_text())
        logger.info(f"Loaded {len(data)} eval samples from {path}")
        return data

    # ── Citation accuracy ─────────────────────────────────────────────────────

    @staticmethod
    def _citation_accuracy(
        answer: str, sources: list[dict[str, Any]]
    ) -> float:
        """
        Lightweight citation accuracy check.

        Checks what fraction of inline citations [N] in the answer
        correspond to actually-retrieved sources.
        """
        import re
        cited_indices = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
        if not cited_indices:
            return 1.0  # No citations → no incorrect citations

        valid_indices = set(range(1, len(sources) + 1))
        correct = cited_indices & valid_indices
        return len(correct) / len(cited_indices) if cited_indices else 1.0

    # ── RAGAS evaluation ──────────────────────────────────────────────────────

    @staticmethod
    def _run_ragas(
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict[str, list[float]]:
        """Run RAGAS metrics; return a dict of metric → per-sample scores."""
        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        try:
            result = evaluate(
                dataset,
                metrics=[
                    answer_faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
            df = result.to_pandas()
            return {
                "faithfulness": df["faithfulness"].tolist(),
                "answer_relevancy": df["answer_relevancy"].tolist(),
                "context_precision": df["context_precision"].tolist(),
                "context_recall": df["context_recall"].tolist(),
            }
        except Exception as exc:
            logger.error(f"RAGAS evaluation failed: {exc}")
            n = len(questions)
            return {
                "faithfulness": [0.0] * n,
                "answer_relevancy": [0.0] * n,
                "context_precision": [0.0] * n,
                "context_recall": [0.0] * n,
            }

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        dataset_path: Path | str | None = None,
        max_samples: int | None = None,
    ) -> EvalReport:
        """
        Execute the full eval harness.

        Parameters
        ----------
        dataset_path : Path or str
            Path to a JSON QA dataset.
        max_samples : int, optional
            Limit the number of samples (useful for quick smoke tests).
        """
        dataset_path = Path(dataset_path or settings.eval_dataset_path)
        samples_data = self.load_dataset(dataset_path)
        if max_samples:
            samples_data = samples_data[:max_samples]

        # ── Step 1: Run pipeline on each question ────────────────────────────
        responses: list[RAGResponse] = []
        logger.info(f"Running pipeline on {len(samples_data)} questions…")
        for item in samples_data:
            resp = self.pipeline.query(item["question"])
            responses.append(resp)

        # ── Step 2: RAGAS scoring ────────────────────────────────────────────
        questions = [d["question"] for d in samples_data]
        ground_truths = [d["ground_truth"] for d in samples_data]
        answers = [r.answer for r in responses]
        contexts = [[s.get("snippet", "") for s in r.sources] for r in responses]

        logger.info("Running RAGAS metrics…")
        ragas_scores = self._run_ragas(questions, answers, contexts, ground_truths)

        # ── Step 3: Assemble EvalSamples ─────────────────────────────────────
        eval_samples = []
        for i, (data, resp) in enumerate(zip(samples_data, responses)):
            sample = EvalSample(
                question=data["question"],
                ground_truth=data["ground_truth"],
                answer=resp.answer,
                contexts=contexts[i],
                sources=resp.sources,
                faithfulness=ragas_scores["faithfulness"][i],
                answer_relevancy=ragas_scores["answer_relevancy"][i],
                context_precision=ragas_scores["context_precision"][i],
                context_recall=ragas_scores["context_recall"][i],
                citation_accuracy=self._citation_accuracy(resp.answer, resp.sources),
                latency_s=resp.latency_s,
                from_cache=resp.from_cache,
            )
            eval_samples.append(sample)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report = EvalReport(samples=eval_samples, timestamp=timestamp)
        report.compute_summary()
        report.print_summary()
        return report
