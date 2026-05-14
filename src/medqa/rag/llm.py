"""Groq LLM wrapper for answer generation and faithfulness judging."""

from __future__ import annotations

import os
import re
import time

from groq import Groq, RateLimitError

MODEL             = "llama-3.3-70b-versatile"
MAX_TOKENS        = 512
MAX_RETRIES       = 4          # retry up to 4 times on 429
RETRY_BASE_DELAY  = 10         # seconds — doubles each retry
# Word-limits applied before sending context to the LLM (keeps token spend low)
ANSWER_CONTEXT_WORDS = 400     # max words in RAG answer-generation context
JUDGE_CONTEXT_WORDS  = 200     # max words in faithfulness-judge context


class LLMClient:
    """Thin wrapper around the Groq Chat Completions API."""

    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def _call(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        """Call the API with exponential-backoff retry on rate-limit errors."""
        delay = RETRY_BASE_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content.strip()
            except RateLimitError as e:
                if attempt == MAX_RETRIES:
                    raise
                print(f"\n  [Rate limit] waiting {delay}s before retry {attempt}/{MAX_RETRIES}…")
                time.sleep(delay)
                delay *= 2

    # ── Answer generation ──────────────────────────────────────────────────

    def answer_baseline(self, question: str) -> str:
        prompt = (
            "You are a medical information assistant. "
            "Answer the following question using your knowledge.\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, factual answer in 2-4 sentences."
        )
        return self._call(prompt)

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        words = text.split()
        return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")

    def answer_rag(self, question: str, context: str) -> str:
        context = self._truncate(context, ANSWER_CONTEXT_WORDS)
        prompt = (
            "You are a medical information assistant. "
            "Use the provided context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the context. Be concise and factual."
        )
        return self._call(prompt)

    def answer_sentence_evidence(self, question: str, evidence: str) -> str:
        prompt = (
            "You are a medical information assistant. "
            "Answer the question using only the specific evidence sentences provided.\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the evidence above."
        )
        return self._call(prompt)

    def answer_uncertainty(
        self, question: str, context: str, confidence: str
    ) -> str:
        context = self._truncate(context, ANSWER_CONTEXT_WORDS)
        prompt = (
            "You are a medical information assistant. "
            f"Context confidence: {confidence}.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer based on the context. "
            "If confidence is Low, explicitly note uncertainty in your answer."
        )
        return self._call(prompt)

    def answer_combined(
        self, question: str, ranked_evidence: str, confidence: str
    ) -> str:
        ranked_evidence = self._truncate(ranked_evidence, ANSWER_CONTEXT_WORDS)
        prompt = (
            "You are a medical information assistant. "
            "Answer the question using the ranked evidence below.\n\n"
            f"Ranked Evidence (by relevance):\n{ranked_evidence}\n\n"
            f"Overall Confidence: {confidence}\n\n"
            f"Question: {question}\n\n"
            "Provide a comprehensive, evidence-based answer. "
            "Note any uncertainty where applicable."
        )
        return self._call(prompt)

    # ── Metric judge ───────────────────────────────────────────────────────

    def judge_faithfulness(
        self, question: str, context: str, answer: str
    ) -> float:
        """Rate how faithfully the answer is grounded in the context (0–1).

        Context is truncated to JUDGE_CONTEXT_WORDS to stay within token budgets.
        """
        # Truncate context to reduce token spend
        words = context.split()
        if len(words) > JUDGE_CONTEXT_WORDS:
            context = " ".join(words[:JUDGE_CONTEXT_WORDS]) + "…"

        prompt = (
            "Rate faithfulness of the answer to the context: 0.0 to 1.0.\n"
            "1.0 = fully supported by context. 0.0 = contradicts context.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            "Reply with ONE decimal number only."
        )
        raw = self._call(prompt, max_tokens=10)
        match = re.search(r"[01]?\.\d+|[01]", raw)
        return float(match.group()) if match else 0.5
