"""Shared metric computation for all RAG experiments."""
from __future__ import annotations

import re

import numpy as np

from .embedder import Embedder
from .llm_client import LLMClient

RELEVANCE_THRESHOLD = 0.40

_W = dict(faith=0.30, ctx_rel=0.20, ctx_pre=0.20, ans_rel=0.30)
_W_NORAG = dict(faith=0.50, ans_rel=0.50)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def extract_sentence_evidence(question: str, chunks: list[str],
                               embedder: Embedder, top_n: int = 5) -> str:
    """Extract the top-n most relevant sentences from retrieved chunks."""
    sentences = []
    for chunk in chunks:
        sentences.extend(re.split(r"(?<=[.!?])\s+", chunk.strip()))
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    if not sentences:
        return " ".join(chunks)
    q_emb  = embedder.encode(question)[0]
    scores = [_cosine(q_emb, embedder.encode(s)[0]) for s in sentences]
    ranked = sorted(zip(scores, sentences), reverse=True)
    return "\n".join(f"- {s}" for _, s in ranked[:top_n])


def compute_metrics(
    question: str,
    answer: str,
    ground_truth: str,
    retrieved_chunks: list[str],
    retrieval_scores: list[float],
    embedder: Embedder,
    llm: LLMClient,
    has_rag: bool,
) -> dict[str, float]:
    """Compute faithfulness, context relevance/precision, answer relevance,
    hallucination rate, and trust score for one question-answer pair."""
    q_emb = embedder.encode(question)[0]
    a_emb = embedder.encode(answer)[0]
    answer_relevance = max(0.0, _cosine(q_emb, a_emb))

    if has_rag and retrieved_chunks:
        c_embs             = embedder.encode(retrieved_chunks)
        sims               = [max(0.0, _cosine(q_emb, e)) for e in c_embs]
        context_relevance  = float(np.mean(sims))
        context_precision  = sum(1 for s in retrieval_scores
                                 if s >= RELEVANCE_THRESHOLD) / len(retrieval_scores)
    else:
        context_relevance = context_precision = 0.0

    context_ref  = "\n\n".join(retrieved_chunks[:3]) if has_rag and retrieved_chunks \
                   else ground_truth
    faithfulness = llm.judge_faithfulness(question, context_ref, answer)

    hallucination_rate = round(1.0 - faithfulness, 4)
    trust_score = round(
        (_W["faith"] * faithfulness + _W["ctx_rel"] * context_relevance
         + _W["ctx_pre"] * context_precision + _W["ans_rel"] * answer_relevance)
        if has_rag
        else (_W_NORAG["faith"] * faithfulness + _W_NORAG["ans_rel"] * answer_relevance),
        4,
    )

    return {
        "faithfulness":       round(faithfulness, 4),
        "context_relevance":  round(context_relevance, 4),
        "context_precision":  round(context_precision, 4),
        "answer_relevance":   round(answer_relevance, 4),
        "hallucination_rate": hallucination_rate,
        "trust_score":        trust_score,
    }
