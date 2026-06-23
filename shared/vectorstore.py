"""Shared ChromaDB in-memory vector store backed by all-MiniLM-L6-v2."""
from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

MODEL_NAME  = "all-MiniLM-L6-v2"
COLLECTION  = "medquad_kb"


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.Client()
        self._ef     = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
        self._col    = self._client.get_or_create_collection(
            name=COLLECTION, embedding_function=self._ef
        )

    _BATCH_SIZE = 5000

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add chunks in batches to respect ChromaDB's max batch size limit."""
        for start in range(0, len(chunks), self._BATCH_SIZE):
            batch = chunks[start : start + self._BATCH_SIZE]
            self._col.add(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {k: c.get(k, "") for k in
                     ("document_id", "title", "source", "medical_topic", "source_link")}
                    for c in batch
                ],
            )

    def query(self, query_text: str, n_results: int = 3) -> list[dict]:
        res = self._col.query(query_texts=[query_text], n_results=n_results)
        return [
            {
                "text":     res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score":    max(0.0, 1.0 - res["distances"][0][i] / 2.0),
            }
            for i in range(len(res["documents"][0]))
        ]

    @property
    def count(self) -> int:
        return self._col.count()
