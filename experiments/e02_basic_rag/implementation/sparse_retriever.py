"""Retrieve via BM25 keyword matching — no embeddings required."""
from __future__ import annotations

from shared.bm25_store import BM25Store
from .config import TOP_K


def retrieve_sparse(question: str, store: BM25Store) -> tuple[list[str], list[float]]:
    hits  = store.query(question, n_results=TOP_K)
    chunks = [h["text"] for h in hits]
    raw    = [h["score"] for h in hits]
    max_s  = max(raw) if raw and max(raw) > 0 else 1.0
    scores = [round(s / max_s, 4) for s in raw]
    return chunks, scores
