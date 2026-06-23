"""E02 — Basic RAG pipeline: retrieve → generate → evaluate."""
from __future__ import annotations
from shared.llm_client import LLMClient
from shared.embedder   import Embedder
from shared.metrics    import compute_metrics
from .ingestion        import build_vector_store
from .retriever        import retrieve
from .generator        import generate


class BasicRAGPipeline:
    def __init__(self) -> None:
        self.llm          = LLMClient()
        self.embedder     = Embedder()
        self.vector_store = build_vector_store()

    def run(self, question: str, ground_truth: str = "") -> dict:
        chunks, scores = retrieve(question, self.vector_store)
        answer         = generate(question, chunks, self.llm)
        metrics        = compute_metrics(
            question=question, answer=answer, ground_truth=ground_truth,
            retrieved_chunks=chunks, retrieval_scores=scores,
            embedder=self.embedder, llm=self.llm, has_rag=True,
        )
        return {"question": question, "answer": answer,
                "retrieved_chunks": chunks, **metrics}
