import re
from typing import Tuple

import pandas as pd

from .config import SOURCE_URL_MAP
from .text import build_dedup_key, extract_url, normalize_question, normalize_text, resolve_source_url

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

    # Remove rows with missing focus_area
    df = df[df["focus_area"].str.len() > 0]

    # Deduplicate per (focus_area, normalized_question) — keep longest answer
    df["normalized_question"] = df["question"].map(normalize_question)
    df = df.sort_values("answer", key=lambda s: s.str.len(), ascending=False)
    df = df.drop_duplicates(subset=["focus_area", "normalized_question"], keep="first")

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 2 – Website link filter
# ---------------------------------------------------------------------------

def filter_url_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that contain a literal http/https URL in the answer text.
    Saved as an intermediate reference file — not the golden sampling pool."""
    df = df.copy()
    df["context_source_id"] = df["answer"].apply(extract_url)
    df = df[df["context_source_id"].str.len() > 0]
    df = df.reset_index(drop=True)
    return df


def resolve_website_links(df: pd.DataFrame) -> pd.DataFrame:
    """Populate context_source_id for every row.

    Priority per row:
      1. Literal URL extracted from the answer text.
      2. Authoritative website URL looked up from the source institution name.

    Rows whose source is not in SOURCE_URL_MAP and whose answer contains no
    URL are dropped (kept only if a website link can be resolved).
    """
    df = df.copy()
    df["context_source_id"] = df.apply(
        lambda r: resolve_source_url(r["answer"], r["source"], SOURCE_URL_MAP),
        axis=1,
    )
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
    target_rows: int = 100,
    rows_per_area: int = ROWS_PER_FOCUS_AREA,
    random_state: int = 42,
) -> pd.DataFrame:
    """Collect exactly target_rows UNIQUE-question rows from the top focus areas.

    Iterates through focus areas (most rows first). For each area:
      1. Pick one row per question_type (rarest type first — diversity pass).
      2. Fill remaining slots from leftover rows, no replacement.
      3. Cap at min(available_unique_rows, rows_per_area).
    Stops as soon as the running total reaches target_rows.
    """
    # Sort areas by available unique-question row count, descending
    area_counts = df["focus_area"].value_counts()
    areas_in_order = area_counts.index.tolist()

    parts: list[pd.DataFrame] = []
    collected = 0

    for area in areas_in_order:
        if collected >= target_rows:
            break

        area_df = df[df["focus_area"] == area].copy()
        need = min(len(area_df), rows_per_area, target_rows - collected)

        # Pass 1: one row per question_type (rarest first)
        type_order = (
            area_df["question_type"].value_counts().index.tolist()[::-1]
        )
        selected_idx: list = []
        for qtype in type_order:
            candidates = area_df[
                (area_df["question_type"] == qtype)
                & (~area_df.index.isin(selected_idx))
            ]
            if not candidates.empty:
                selected_idx.append(
                    candidates.sample(1, random_state=random_state).index[0]
                )
            if len(selected_idx) >= need:
                break

        # Pass 2: fill remaining slots without replacement
        if len(selected_idx) < need:
            remaining = area_df[~area_df.index.isin(selected_idx)]
            extra = remaining.sample(
                min(need - len(selected_idx), len(remaining)),
                random_state=random_state,
            )
            selected_idx.extend(extra.index.tolist())

        parts.append(area_df.loc[selected_idx[:need]])
        collected += need

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
