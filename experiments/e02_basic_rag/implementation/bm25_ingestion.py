"""Build a BM25 keyword index over the knowledge base chunks."""
from __future__ import annotations

import pandas as pd

from preprocessing.config import KNOWLEDGE_BASE_CSV
from shared.bm25_store import BM25Store
from shared.chunker import chunk_documents
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def build_bm25_store() -> BM25Store:
    kb     = pd.read_csv(KNOWLEDGE_BASE_CSV).to_dict(orient="records")
    chunks = chunk_documents(kb, CHUNK_SIZE, CHUNK_OVERLAP)
    return BM25Store(chunks)
