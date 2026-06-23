from __future__ import annotations
from shared.vectorstore import VectorStore
from shared.embedder    import Embedder
from shared.metrics     import extract_sentence_evidence
from .config import TOP_K, TOP_N_SENTENCES, HIGH_CONF_THRESHOLD, LOW_CONF_THRESHOLD, QUERY_EXPANSIONS

def _expand_query(question: str) -> str:
    q = question.lower()
    extra = [synonyms for key, synonyms in QUERY_EXPANSIONS.items() if key in q]
    return question + (" " + " ".join(extra) if extra else "")

def retrieve_combined(question: str, vs: VectorStore, embedder: Embedder):
    expanded = _expand_query(question)
    hits     = vs.query(expanded, n_results=TOP_K)
    chunks   = [h["text"] for h in hits]
    scores   = [h["score"] for h in hits]
    top      = max(scores) if scores else 0.0
    confidence = "High" if top >= HIGH_CONF_THRESHOLD else ("Medium" if top >= LOW_CONF_THRESHOLD else "Low")
    evidence = extract_sentence_evidence(question, chunks, embedder, TOP_N_SENTENCES)
    ranked   = f"[Confidence: {confidence}]\n{evidence}"
    return chunks, scores, ranked, confidence
