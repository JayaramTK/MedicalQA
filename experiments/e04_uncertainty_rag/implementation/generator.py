from __future__ import annotations
from shared.llm_client import LLMClient

def generate(question: str, chunks: list[str], confidence: str, llm: LLMClient) -> str:
    return llm.answer_uncertainty(question, "\n\n".join(chunks), confidence)
