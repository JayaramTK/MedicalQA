import re
import unicodedata

QUESTION_PREFIXES = ["what is", "what are", "who is", "who are", "how", "when", "where", "why", "does", "do", "is", "are", "can", "could", "should", "will", "which"]


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_question(question: str) -> str:
    text = normalize_text(question).lower()
    text = re.sub(r"["'“”‘’\[\]]", "", text)
    text = re.sub(r"\?+$", "", text)
    text = text.strip()
    if not text:
        return text
    for prefix in QUESTION_PREFIXES:
        if text.startswith(prefix + " "):
            return text
    return text


def build_dedup_key(row: dict) -> str:
    question = normalize_question(row.get("question", ""))
    answer = normalize_text(row.get("answer", ""))
    return f"{question}|{answer}"
