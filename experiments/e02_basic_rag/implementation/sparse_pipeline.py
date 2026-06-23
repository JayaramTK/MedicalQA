"""E02.1 — Sparse RAG pipeline: BM25 retrieval → LLM generation → metrics."""
from __future__ import annotations

from shared.embedder   import Embedder
from shared.llm_client import LLMClient
from shared.metrics    import compute_metrics
from .bm25_ingestion   import build_bm25_store
from .sparse_retriever import retrieve_sparse
from .generator        import generate


class SparseRAGPipeline:
    def __init__(self) -> None:
        self.llm      = LLMClient()
        self.embedder = Embedder()
        self.store    = build_bm25_store()

    def run(self, question: str, ground_truth: str = "") -> dict:
        chunks, scores = retrieve_sparse(question, self.store)
        answer         = generate(question, chunks, self.llm)
        metrics        = compute_metrics(
            question=question, answer=answer, ground_truth=ground_truth,
            retrieved_chunks=chunks, retrieval_scores=scores,
            embedder=self.embedder, llm=self.llm, has_rag=True,
        )
        return {"question": question, "answer": answer,
                "retrieved_chunks": chunks, **metrics}
