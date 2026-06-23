"""E02 — Ingestion: load knowledge base and populate ChromaDB collection."""
from __future__ import annotations
import pandas as pd
from preprocessing.config import KNOWLEDGE_BASE_CSV
from shared.chunker import chunk_documents
from shared.vectorstore import VectorStore
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def build_vector_store() -> VectorStore:
    kb     = pd.read_csv(KNOWLEDGE_BASE_CSV).to_dict(orient="records")
    chunks = chunk_documents(kb, CHUNK_SIZE, CHUNK_OVERLAP)
    vs     = VectorStore()
    vs.add_chunks(chunks)
    return vs
