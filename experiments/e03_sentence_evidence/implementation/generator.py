from __future__ import annotations
from shared.llm_client import LLMClient

def generate(question: str, evidence: str, llm: LLMClient) -> str:
    return llm.answer_sentence_evidence(question, evidence)
