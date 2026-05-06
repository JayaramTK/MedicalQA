import re
from typing import Tuple

import pandas as pd

from .text import build_dedup_key, extract_url, normalize_question, normalize_text

# Difficulty thresholds (word counts)
_Q_SHORT = 10   # <= short question
_Q_LONG = 20    # > long question
_A_SHORT = 200  # <= short answer
_A_LONG = 400   # > long answer

MIN_ANSWER_WORDS = 100
TOP_N_FOCUS_AREAS = 10
ROWS_PER_FOCUS_AREA = 10

# Question-type keyword patterns (order matters — most specific first)
_QTYPE_PATTERNS = [
    ("symptom",    re.compile(r'\b(symptom|symptoms|sign|signs)\b', re.I)),
    ("treatment",  re.compile(r'\b(treatment|treatments|treat|therapy|medication|drug|cure|manage)\b', re.I)),
    ("cause",      re.compile(r'\b(cause|causes|caused|why|etiology)\b', re.I)),
    ("Diagnosis",  re.compile(r'\b(diagnos\w*|detect|test|screening|examination)\b', re.I)),
    ("Prevention", re.compile(r'\b(prevent|prevention|avoid|reducing risk)\b', re.I)),
]


# ---------------------------------------------------------------------------
# Step 1 – Basic cleaning
# ---------------------------------------------------------------------------

def clean_medquad(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all basic cleaning rules and return a reset-indexed DataFrame."""
    df = df.copy()

    # Normalize text fields
    for col in ("question", "answer", "source", "focus_area"):
        df[col] = df[col].fillna("").map(normalize_text)

    # Remove rows where question or answer is missing
    df = df[df["question"].str.len() > 0]
    df = df[df["answer"].str.len() > 0]

    # Remove very short answers (< 100 words)
    df = df[df["answer"].str.split().str.len() >= MIN_ANSWER_WORDS]

    # Remove noise / incomplete text: answer must be ≥70% alphabetic chars
    def _alpha_ratio(text: str) -> float:
        letters = sum(c.isalpha() or c.isspace() for c in text)
        return letters / max(len(text), 1)

    df = df[df["answer"].apply(_alpha_ratio) >= 0.70]

    # Remove duplicate questions (keep first occurrence)
    df["normalized_question"] = df["question"].map(normalize_question)
    df = df.drop_duplicates(subset=["normalized_question"], keep="first")

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 2 – URL filter
# ---------------------------------------------------------------------------

def filter_url_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that contain a valid http/https URL in the answer.
    Extracts the first URL as context_source_id."""
    df = df.copy()
    df["context_source_id"] = df["answer"].apply(extract_url)
    df = df[df["context_source_id"].str.len() > 0]
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 3 – Enrichment: question_type and difficulty_level
# ---------------------------------------------------------------------------

def _classify_question_type(question: str) -> str:
    for label, pattern in _QTYPE_PATTERNS:
        if pattern.search(question):
            return label
    return "general"


def _classify_difficulty(question: str, answer: str) -> str:
    q_words = len(question.split())
    a_words = len(answer.split())
    if q_words > _Q_LONG or a_words > _A_LONG:
        return "hard"
    if q_words <= _Q_SHORT and a_words <= _A_SHORT:
        return "easy"
    return "medium"


def enrich_medquad(df: pd.DataFrame) -> pd.DataFrame:
    """Add question_type and difficulty_level columns."""
    df = df.copy()
    df["question_type"] = df["question"].apply(_classify_question_type)
    df["difficulty_level"] = df.apply(
        lambda r: _classify_difficulty(r["question"], r["answer"]), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# Step 4 – Golden dataset sampling
# ---------------------------------------------------------------------------

def sample_golden_dataset(
    df: pd.DataFrame,
    top_n: int = TOP_N_FOCUS_AREAS,
    rows_per_area: int = ROWS_PER_FOCUS_AREA,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample rows_per_area rows from each of the top_n focus areas,
    maximising question_type diversity within each area."""

    top_areas = (
        df["focus_area"].value_counts().head(top_n).index.tolist()
    )
    filtered = df[df["focus_area"].isin(top_areas)].copy()

    parts = []
    for area in top_areas:
        area_df = filtered[filtered["focus_area"] == area]

        # One row per question_type (for diversity) — avoid groupby.apply NaN bug
        diverse_rows = []
        for _, group in area_df.groupby("question_type", sort=False):
            diverse_rows.append(group.sample(1, random_state=random_state))
        diverse = pd.concat(diverse_rows) if diverse_rows else area_df.iloc[:0]

        # Fill up to rows_per_area if we have more data
        if len(diverse) < rows_per_area:
            remaining = area_df.loc[~area_df.index.isin(diverse.index)]
            still_needed = rows_per_area - len(diverse)
            if len(remaining) >= still_needed:
                extra = remaining.sample(still_needed, random_state=random_state)
            else:
                extra = area_df.sample(
                    still_needed, replace=True, random_state=random_state
                )
            diverse = pd.concat([diverse, extra])

        parts.append(diverse.head(rows_per_area))

    sampled = pd.concat(parts).reset_index(drop=True)

    # Build final golden columns
    sampled.insert(0, "golden_id", ["GOLD{:04d}".format(i + 1) for i in range(len(sampled))])
    sampled["ground_truth"] = sampled["answer"]
    sampled["expected_context"] = sampled["answer"]

    return sampled


# ---------------------------------------------------------------------------
# Legacy helpers (used by build_knowledge_db.py and tests)
# ---------------------------------------------------------------------------

def transform_medquad(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper: clean + dedup with dedup_key."""
    df = clean_medquad(df)
    df["dedup_key"] = df.apply(lambda row: build_dedup_key(row), axis=1)
    df = df.drop_duplicates(subset=["dedup_key"], keep="first")
    df = df.drop(columns=["dedup_key"])
    df = df.reset_index(drop=True)
    return df


def build_summary(df: pd.DataFrame) -> Tuple[int, int, int]:
    original = len(df)
    filtered = df[df["question"].astype(bool) & df["answer"].astype(bool)]
    cleaned = len(filtered.drop_duplicates(subset=["question", "answer"]))
    return original, len(filtered), cleaned
