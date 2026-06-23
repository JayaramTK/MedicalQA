from __future__ import annotations
from shared.vectorstore import VectorStore
from .config import TOP_K, HIGH_CONF_THRESHOLD, LOW_CONF_THRESHOLD

def retrieve_with_confidence(question: str, vs: VectorStore) -> tuple[list[str], list[float], str]:
    hits   = vs.query(question, n_results=TOP_K)
    chunks = [h["text"] for h in hits]
    scores = [h["score"] for h in hits]
    top    = max(scores) if scores else 0.0
    if top >= HIGH_CONF_THRESHOLD:
        confidence = "High"
    elif top >= LOW_CONF_THRESHOLD:
        confidence = "Medium"
    else:
        confidence = "Low"
    return chunks, scores, confidence
