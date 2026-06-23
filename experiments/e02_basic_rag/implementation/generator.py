"""E02 — Generator: produce answer from retrieved context using the LLM."""
from __future__ import annotations
from shared.llm_client import LLMClient


def generate(question: str, chunks: list[str], llm: LLMClient) -> str:
    context = "\n\n".join(chunks)
    return llm.answer_rag(question, context)
