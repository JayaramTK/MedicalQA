"""E02.2 — Hybrid RAG pipeline: Dense + BM25 → RRF fusion → LLM → metrics."""
from __future__ import annotations

from shared.embedder    import Embedder
from shared.llm_client  import LLMClient
from shared.metrics     import compute_metrics
from shared.vectorstore import VectorStore
from .bm25_ingestion    import build_bm25_store
from .ingestion         import build_vector_store
from .hybrid_retriever  import retrieve_hybrid
from .generator         import generate


class HybridRAGPipeline:
    def __init__(self) -> None:
        self.llm          = LLMClient()
        self.embedder     = Embedder()
        self.dense_store  = build_vector_store()
        self.sparse_store = build_bm25_store()

    def run(self, question: str, ground_truth: str = "") -> dict:
        chunks, scores = retrieve_hybrid(question, self.dense_store, self.sparse_store)
        answer         = generate(question, chunks, self.llm)
        metrics        = compute_metrics(
            question=question, answer=answer, ground_truth=ground_truth,
            retrieved_chunks=chunks, retrieval_scores=scores,
            embedder=self.embedder, llm=self.llm, has_rag=True,
        )
        return {"question": question, "answer": answer,
                "retrieved_chunks": chunks, **metrics}
