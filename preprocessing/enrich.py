"""
Enrich cleaned data.

  - Resolve context_source_id: literal URL from answer → source institution URL
  - Classify question_type by keyword matching
  - Classify difficulty_level by word-count thresholds
  - Save URL-filtered reference file (rows with a literal URL in answer text)
"""
import re

import pandas as pd

from .config import SOURCE_URL_MAP, URL_FILTERED_CSV_PATH

_Q_SHORT, _Q_LONG = 10, 20
_A_SHORT, _A_LONG = 200, 400

_QTYPE_PATTERNS = [
    ("symptom",    re.compile(r"\b(symptom|symptoms|sign|signs)\b", re.I)),
    ("treatment",  re.compile(r"\b(treatment|treat|therapy|medication|drug|cure|manage)\b", re.I)),
    ("cause",      re.compile(r"\b(cause|causes|caused|why|etiology)\b", re.I)),
    ("Diagnosis",  re.compile(r"\b(diagnos\w*|detect|test|screening|examination)\b", re.I)),
    ("Prevention", re.compile(r"\b(prevent|prevention|avoid|reducing risk)\b", re.I)),
]

_URL_RE = re.compile(r"https?://[^\s\",)\]]+")


def _extract_url(text: str) -> str:
    m = _URL_RE.search(str(text))
    return m.group(0).rstrip(".,;:") if m else ""


def _resolve_source_url(answer: str, source: str) -> str:
    url = _extract_url(answer)
    return url if url else SOURCE_URL_MAP.get(source, "")


def _question_type(question: str) -> str:
    for label, pattern in _QTYPE_PATTERNS:
        if pattern.search(question):
            return label
    return "general"


def _difficulty(question: str, answer: str) -> str:
    qw, aw = len(question.split()), len(answer.split())
    if qw > _Q_LONG or aw > _A_LONG:
        return "hard"
    if qw <= _Q_SHORT and aw <= _A_SHORT:
        return "easy"
    return "medium"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["context_source_id"] = df.apply(
        lambda r: _resolve_source_url(r["answer"], r["source"]), axis=1
    )
    df = df[df["context_source_id"].str.len() > 0].reset_index(drop=True)
    df["question_type"]    = df["question"].apply(_question_type)
    df["difficulty_level"] = df.apply(lambda r: _difficulty(r["question"], r["answer"]), axis=1)
    print(f"  [Enrich] {len(df):,} rows with website links resolved")
    return df


def save_url_filtered(df: pd.DataFrame) -> None:
    literal = df[df["answer"].apply(_extract_url).str.len() > 0].copy()
    literal["context_source_id"] = literal["answer"].apply(_extract_url)
    URL_FILTERED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    literal.to_csv(URL_FILTERED_CSV_PATH, index=False)
    print(f"  [Enrich] URL-filtered reference ({len(literal)} rows) → {URL_FILTERED_CSV_PATH}")


if __name__ == "__main__":
    from .extract import extract
    from .clean import clean
    df = enrich(clean(extract()))
    save_url_filtered(df)
