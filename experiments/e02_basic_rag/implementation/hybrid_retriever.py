"""
Hybrid retriever — Dense (ChromaDB) + Sparse (BM25) merged via
Reciprocal Rank Fusion (RRF).

Each chunk gets a score of  1 / (RRF_K + rank + 1)  from each ranker.
Scores are summed across both lists and the top-K chunks are returned.
"""
from __future__ import annotations

from shared.bm25_store  import BM25Store
from shared.vectorstore import VectorStore
from .config import TOP_K, RRF_K


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def retrieve_hybrid(
    question: str,
    dense_store: VectorStore,
    sparse_store: BM25Store,
) -> tuple[list[str], list[float]]:
    """Merge dense and sparse rankings and return top-K fused results."""
    dense_hits  = dense_store.query(question, n_results=TOP_K)
    sparse_hits = sparse_store.query(question, n_results=TOP_K)

    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, str]   = {}

    for rank, hit in enumerate(dense_hits):
        key = hit["text"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + _rrf(rank)
        chunk_map[key]  = hit["text"]

    for rank, hit in enumerate(sparse_hits):
        key = hit["text"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + _rrf(rank)
        chunk_map[key]  = hit["text"]

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    chunks = [chunk_map[k] for k, _ in ranked]
    scores = [round(s, 4) for _, s in ranked]
    return chunks, scores
