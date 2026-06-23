from __future__ import annotations
from shared.llm_client import LLMClient
from shared.embedder   import Embedder
from shared.metrics    import compute_metrics
from .ingestion        import build_vector_store
from .retriever        import retrieve_with_confidence
from .generator        import generate

class UncertaintyRAGPipeline:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.embedder = Embedder()
        self.vector_store = build_vector_store()

    def run(self, question: str, ground_truth: str = "") -> dict:
        chunks, scores, confidence = retrieve_with_confidence(question, self.vector_store)
        answer  = generate(question, chunks, confidence, self.llm)
        metrics = compute_metrics(
            question=question, answer=answer, ground_truth=ground_truth,
            retrieved_chunks=chunks, retrieval_scores=scores,
            embedder=self.embedder, llm=self.llm, has_rag=True,
        )
        return {"question": question, "answer": answer,
                "confidence": confidence, **metrics}
