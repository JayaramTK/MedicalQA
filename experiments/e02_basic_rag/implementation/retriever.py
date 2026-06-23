"""E02 — Retriever: semantic similarity search via ChromaDB."""
from __future__ import annotations
from shared.vectorstore import VectorStore
from .config import TOP_K


def retrieve(question: str, vs: VectorStore) -> tuple[list[str], list[float]]:
    hits   = vs.query(question, n_results=TOP_K)
    chunks = [h["text"] for h in hits]
    scores = [h["score"] for h in hits]
    return chunks, scores
