from __future__ import annotations
from shared.vectorstore import VectorStore
from shared.embedder    import Embedder
from shared.metrics     import extract_sentence_evidence
from .config import TOP_K, TOP_N_SENTENCES

def retrieve_evidence(question: str, vs: VectorStore, embedder: Embedder) -> tuple[list[str], list[float], str]:
    hits     = vs.query(question, n_results=TOP_K)
    chunks   = [h["text"] for h in hits]
    scores   = [h["score"] for h in hits]
    evidence = extract_sentence_evidence(question, chunks, embedder, TOP_N_SENTENCES)
    return chunks, scores, evidence
