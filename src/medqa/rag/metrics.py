"""Metric computation for RAG experiments."""

from __future__ import annotations

import re

import numpy as np

from .embedder import Embedder
from .llm import LLMClient

# Threshold above which a retrieved chunk is considered "relevant"
RELEVANCE_THRESHOLD = 0.40

# Trust score weights
_W_FAITH   = 0.30
_W_CTX_REL = 0.20
_W_CTX_PRE = 0.20
_W_ANS_REL = 0.30

# Weights when no context exists (E01)
_W_FAITH_NORAG   = 0.50
_W_ANS_REL_NORAG = 0.50


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def extract_sentence_evidence(
    question: str, chunks: list[str], embedder: Embedder, top_n: int = 5
) -> str:
    """Return the top-n most relevant sentences from the retrieved chunks."""
    sentences: list[str] = []
    for chunk in chunks:
        sentences.extend(re.split(r"(?<=[.!?])\s+", chunk.strip()))

    sentences = [s.strip() for s in sentences if len(s.split()) > 5]
    if not sentences:
        return " ".join(chunks)

    q_emb = embedder.encode(question)[0]
    s_embs = embedder.encode(sentences)
    scores = [_cosine(q_emb, s_emb) for s_emb in s_embs]

    ranked = sorted(zip(scores, sentences), reverse=True)
    top = [s for _, s in ranked[:top_n]]
    return "\n".join(f"- {s}" for s in top)


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
    """Compute all six metrics for one question-answer pair.

    For E01 (has_rag=False) context metrics are 0 and faithfulness
    is judged against the ground-truth text as reference context.

    Returns dict with keys:
        faithfulness, context_relevance, context_precision,
        answer_relevance, hallucination_rate, trust_score
    """
    # ── Embedding-based metrics ───────────────────────────────────────────
    q_emb = embedder.encode(question)[0]
    a_emb = embedder.encode(answer)[0]
    answer_relevance = max(0.0, _cosine(q_emb, a_emb))

    if has_rag and retrieved_chunks:
        c_embs = embedder.encode(retrieved_chunks)
        sims = [max(0.0, _cosine(q_emb, c_emb)) for c_emb in c_embs]
        context_relevance = float(np.mean(sims))
        relevant = sum(1 for s in retrieval_scores if s >= RELEVANCE_THRESHOLD)
        context_precision = relevant / len(retrieval_scores)
    else:
        context_relevance = 0.0
        context_precision = 0.0

    # ── LLM faithfulness judge ────────────────────────────────────────────
    if has_rag and retrieved_chunks:
        context_for_judge = "\n\n".join(retrieved_chunks[:3])
    else:
        context_for_judge = ground_truth   # use ground truth as reference for E01

    faithfulness = llm.judge_faithfulness(question, context_for_judge, answer)

    # ── Derived metrics ───────────────────────────────────────────────────
    hallucination_rate = round(1.0 - faithfulness, 4)

    if has_rag:
        trust_score = (
            _W_FAITH   * faithfulness
            + _W_CTX_REL * context_relevance
            + _W_CTX_PRE * context_precision
            + _W_ANS_REL * answer_relevance
        )
    else:
        trust_score = (
            _W_FAITH_NORAG   * faithfulness
            + _W_ANS_REL_NORAG * answer_relevance
        )

    return {
        "faithfulness":       round(faithfulness, 4),
        "context_relevance":  round(context_relevance, 4),
        "context_precision":  round(context_precision, 4),
        "answer_relevance":   round(answer_relevance, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "trust_score":        round(trust_score, 4),
    }
