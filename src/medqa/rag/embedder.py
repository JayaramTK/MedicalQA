"""Sentence-transformer embedding wrapper."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps all-MiniLM-L6-v2 for encoding and similarity computation."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for two already-normalised vectors."""
        return float(np.dot(a, b))

    def similarity_matrix(
        self, query_emb: np.ndarray, corpus_embs: np.ndarray
    ) -> np.ndarray:
        """Return similarity scores between one query and N corpus embeddings."""
        return corpus_embs @ query_emb
