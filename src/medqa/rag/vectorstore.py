"""ChromaDB vector store wrapper."""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class VectorStore:
    """In-memory ChromaDB collection backed by all-MiniLM-L6-v2."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    COLLECTION = "medquad_kb"

    def __init__(self) -> None:
        self._client = chromadb.Client()
        self._ef = SentenceTransformerEmbeddingFunction(model_name=self.MODEL_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self._ef,
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add chunk dicts (must have chunk_id and text keys)."""
        self._collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "document_id":  c.get("document_id", ""),
                    "title":        c.get("title", ""),
                    "source":       c.get("source", ""),
                    "medical_topic":c.get("medical_topic", ""),
                    "source_link":  c.get("source_link", ""),
                }
                for c in chunks
            ],
        )

    def query(
        self, query_text: str, n_results: int = 3
    ) -> list[dict]:
        """Return top-n chunks with text, metadata, and similarity score."""
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append(
                {
                    "text":     results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    # ChromaDB returns L2 distance; convert to 0-1 similarity
                    "score":    max(0.0, 1.0 - results["distances"][0][i] / 2.0),
                }
            )
        return chunks

    @property
    def count(self) -> int:
        return self._collection.count()
