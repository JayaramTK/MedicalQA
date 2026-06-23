"""
Shared LLM client — Groq (llama-3.3-70b-versatile) with exponential-backoff retry.
All RAG strategies import from here.
"""
from __future__ import annotations

import os
import re
import time

from groq import Groq, RateLimitError

MODEL             = "llama-3.3-70b-versatile"
MAX_TOKENS        = 512
MAX_RETRIES       = 4
RETRY_BASE_DELAY  = 10         # seconds — doubles each attempt
ANSWER_CONTEXT_WORDS = 400     # max words sent as RAG context
JUDGE_CONTEXT_WORDS  = 200     # max words sent to faithfulness judge


class LLMClient:
    """Thin, retry-safe wrapper around the Groq Chat Completions API."""

    def __init__(self, model: str = MODEL) -> None:
        self.model   = model
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # ── Internal call with retry ──────────────────────────────────────────────

    @staticmethod
    def _parse_retry_after(error: RateLimitError) -> float | None:
        """Extract wait seconds from Groq's 'Please try again in Xm Y.Zs' message."""
        m = re.search(r"try again in\s+(?:(\d+)m)?(?:\s*(\d+(?:\.\d+)?)s)?", str(error))
        if not m:
            return None
        minutes = float(m.group(1) or 0)
        seconds = float(m.group(2) or 0)
        total   = minutes * 60 + seconds
        return total if total > 0 else None

    def _call(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        delay = RETRY_BASE_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content.strip()
            except RateLimitError as exc:
                if attempt == MAX_RETRIES:
                    raise
                # Use Groq's suggested wait time if it's longer than our backoff
                parsed = self._parse_retry_after(exc)
                wait   = max(parsed + 5, delay) if parsed else delay
                print(f"\n  [Rate limit] waiting {wait:.0f}s (retry {attempt}/{MAX_RETRIES})…")
                time.sleep(wait)
                delay *= 2

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        words = text.split()
        return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")

    # ── Answer generation methods ─────────────────────────────────────────────

    def answer_baseline(self, question: str) -> str:
        return self._call(
            "You are a medical information assistant. "
            "Answer the following question using your knowledge.\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, factual answer in 2-4 sentences."
        )

    def answer_zero_shot(self, question: str) -> str:
        """Zero-shot: minimal prompt with no role setup or instructions."""
        return self._call(
            f"Answer the following medical question accurately and concisely.\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    def answer_few_shot(self, question: str) -> str:
        """Few-shot: three in-context medical QA examples followed by the query."""
        return self._call(
            "You are a medical information assistant. "
            "Use the examples below as a guide.\n\n"
            "Q: What is Type 1 diabetes?\n"
            "A: Type 1 diabetes is an autoimmune condition where the immune system destroys "
            "insulin-producing beta cells in the pancreas, requiring lifelong insulin therapy.\n\n"
            "Q: What are common symptoms of hypertension?\n"
            "A: Hypertension is often asymptomatic. When symptoms occur they include headaches, "
            "shortness of breath, and in severe cases chest pain or vision changes.\n\n"
            "Q: How is asthma treated?\n"
            "A: Asthma is managed with short-acting bronchodilators for quick relief and inhaled "
            "corticosteroids for long-term control, alongside trigger avoidance.\n\n"
            f"Q: {question}\nA:"
        )

    def answer_cot(self, question: str) -> str:
        """Chain-of-Thought: ask the model to reason step by step before answering."""
        return self._call(
            "You are a medical information assistant. "
            "Think step by step before giving your final answer.\n\n"
            f"Question: {question}\n\n"
            "Reason through this step by step, then state your final answer in 2-3 sentences."
        )

    def answer_tot(self, question: str) -> str:
        """Tree-of-Thought: explore multiple reasoning perspectives, then synthesise."""
        return self._call(
            "You are a medical information assistant. "
            "Consider the following perspectives before answering:\n"
            "1. Clinical / pathophysiological perspective\n"
            "2. Patient-facing / educational perspective\n"
            "3. Evidence-based / treatment perspective\n\n"
            f"Question: {question}\n\n"
            "Briefly consider each perspective, then provide a synthesised answer in 2-4 sentences."
        )

    def answer_self_consistency(self, question: str) -> str:
        """Self-consistency: sample 3 independent answers, synthesise the most consistent one."""
        draft1 = self._call(
            f"You are a medical information assistant. Answer concisely: {question}"
        )
        draft2 = self._call(
            f"You are a medical expert. Provide a factual answer to: {question}"
        )
        draft3 = self._call(
            f"As a healthcare professional, briefly and accurately answer: {question}"
        )
        return self._call(
            "You are a medical information assistant. "
            "Three independent responses to the same question are given below. "
            "Identify the most consistent and accurate conclusion and state it in 2-4 sentences.\n\n"
            f"Response 1: {self._truncate(draft1, 80)}\n"
            f"Response 2: {self._truncate(draft2, 80)}\n"
            f"Response 3: {self._truncate(draft3, 80)}\n\n"
            f"Question: {question}\n\nFinal answer:"
        )

    def answer_rag(self, question: str, context: str) -> str:
        context = self._truncate(context, ANSWER_CONTEXT_WORDS)
        return self._call(
            "You are a medical information assistant. "
            "Use the provided context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the context. Be concise and factual."
        )

    def answer_sentence_evidence(self, question: str, evidence: str) -> str:
        evidence = self._truncate(evidence, ANSWER_CONTEXT_WORDS)
        return self._call(
            "You are a medical information assistant. "
            "Answer the question using only the specific evidence sentences provided.\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the evidence above."
        )

    def answer_uncertainty(self, question: str, context: str, confidence: str) -> str:
        context = self._truncate(context, ANSWER_CONTEXT_WORDS)
        return self._call(
            "You are a medical information assistant. "
            f"Context confidence: {confidence}.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the context. "
            "If confidence is Low, explicitly note uncertainty in your answer."
        )

    def answer_combined(self, question: str, ranked_evidence: str, confidence: str) -> str:
        ranked_evidence = self._truncate(ranked_evidence, ANSWER_CONTEXT_WORDS)
        return self._call(
            "You are a medical information assistant. "
            "Answer the question using the ranked evidence below.\n\n"
            f"Ranked Evidence (by relevance):\n{ranked_evidence}\n\n"
            f"Overall Confidence: {confidence}\n\n"
            f"Question: {question}\n\n"
            "Provide a comprehensive, evidence-based answer. "
            "Note any uncertainty where applicable."
        )

    # ── Metric judge ──────────────────────────────────────────────────────────

    def judge_faithfulness(self, question: str, context: str, answer: str) -> float:
        """Rate how faithfully the answer is grounded in the context (0–1)."""
        words = context.split()
        if len(words) > JUDGE_CONTEXT_WORDS:
            context = " ".join(words[:JUDGE_CONTEXT_WORDS]) + "…"
        raw = self._call(
            "Rate faithfulness of the answer to the context: 0.0 to 1.0.\n"
            "1.0 = fully supported. 0.0 = contradicts context.\n\n"
            f"Context: {context}\nQuestion: {question}\nAnswer: {answer}\n\n"
            "Reply with ONE decimal number only.",
            max_tokens=10,
        )
        match = re.search(r"[01]?\.\d+|[01]", raw)
        return float(match.group()) if match else 0.5
