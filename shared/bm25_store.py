"""
Shared BM25 sparse retrieval store.

Builds a keyword index over knowledge base chunks and retrieves by term
matching — no embeddings required.
"""
from __future__ import annotations

import re
from rank_bm25 import BM25Okapi


def _tokenise(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


class BM25Store:
    """BM25 index over a flat list of chunk dicts."""

    def __init__(self, chunks: list[dict]) -> None:
        self._chunks   = chunks
        self._corpus   = [_tokenise(c["text"]) for c in chunks]
        self._index    = BM25Okapi(self._corpus)

    def query(self, query_text: str, n_results: int = 3) -> list[dict]:
        """Return top-n chunks ranked by BM25 score."""
        tokens = _tokenise(query_text)
        scores = self._index.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:n_results]

        return [
            {
                "text":     self._chunks[i]["text"],
                "metadata": {k: self._chunks[i].get(k, "") for k in
                             ("document_id", "title", "source",
                              "medical_topic", "source_link")},
                "score":    float(score),
                "rank":     pos,
            }
            for pos, (i, score) in enumerate(ranked)
        ]
