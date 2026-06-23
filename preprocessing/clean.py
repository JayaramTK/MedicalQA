"""
Clean raw MedQuAD data.

Rules applied (in order):
  1. Normalise text — Unicode NFKC, collapse whitespace
  2. Drop rows with missing question, answer, or focus_area
  3. Drop answers shorter than MIN_ANSWER_WORDS
  4. Drop answers with < MIN_ALPHA_RATIO alphabetic characters (noise filter)
  5. Deduplicate per (focus_area, normalised_question) — keep longest answer
  6. Reset index
"""
import re
import unicodedata

import pandas as pd

from .config import CLEANED_CSV_PATH, MIN_ALPHA_RATIO, MIN_ANSWER_WORDS


def _normalise(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_question(question: str) -> str:
    text = _normalise(question).lower()
    text = re.sub('["\'\\u201c\\u201d\\u2018\\u2019\\[\\]]', "", text)
    return re.sub(r"\?+$", "", text).strip()


def _alpha_ratio(text: str) -> float:
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters / max(len(text), 1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ("question", "answer", "source", "focus_area"):
        df[col] = df[col].fillna("").map(_normalise)

    df = df[df["question"].str.len() > 0]
    df = df[df["answer"].str.len() > 0]
    df = df[df["focus_area"].str.len() > 0]
    df = df[df["answer"].str.split().str.len() >= MIN_ANSWER_WORDS]
    df = df[df["answer"].apply(_alpha_ratio) >= MIN_ALPHA_RATIO]

    df["normalized_question"] = df["question"].apply(_normalise_question)
    df = df.sort_values("answer", key=lambda s: s.str.len(), ascending=False)
    df = df.drop_duplicates(subset=["focus_area", "normalized_question"], keep="first")

    df = df.reset_index(drop=True)
    print(f"  [Clean] {len(df):,} rows after cleaning")
    return df


def save(df: pd.DataFrame) -> None:
    CLEANED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"  [Clean] Saved → {CLEANED_CSV_PATH}")


if __name__ == "__main__":
    from .extract import extract
    df = clean(extract())
    save(df)
