from __future__ import annotations
from shared.llm_client import LLMClient

def generate(question: str, ranked_evidence: str, confidence: str, llm: LLMClient) -> str:
    return llm.answer_combined(question, ranked_evidence, confidence)
