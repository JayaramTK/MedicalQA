"""E01 — Baseline LLM pipeline. Answers questions with no retrieval."""
from __future__ import annotations

from shared.llm_client import LLMClient
from shared.embedder   import Embedder
from shared.metrics    import compute_metrics


class BaselinePipeline:
    def __init__(self) -> None:
        self.llm      = LLMClient()
        self.embedder = Embedder()

    def run(self, question: str, ground_truth: str = "") -> dict:
        answer  = self.llm.answer_baseline(question)
        metrics = compute_metrics(
            question=question, answer=answer, ground_truth=ground_truth,
            retrieved_chunks=[], retrieval_scores=[],
            embedder=self.embedder, llm=self.llm, has_rag=False,
        )
        return {"question": question, "answer": answer, **metrics}
