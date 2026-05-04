from typing import Tuple

import pandas as pd

from .text import build_dedup_key, normalize_question, normalize_text

MIN_ANSWER_WORDS = 100
TOP_N_FOCUS_AREAS = 10
ROWS_PER_FOCUS_AREA = 10


def sample_golden_dataset(
    df: pd.DataFrame,
    top_n: int = TOP_N_FOCUS_AREAS,
    rows_per_area: int = ROWS_PER_FOCUS_AREA,
    min_answer_words: int = MIN_ANSWER_WORDS,
    random_state: int = 42,
) -> pd.DataFrame:

    # Filter by answer length
    qualified = df[df["answer"].str.split().str.len() >= min_answer_words].copy()

    # Top focus areas
    top_areas = (
        qualified["focus_area"]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )

    filtered = qualified[qualified["focus_area"].isin(top_areas)].copy()

    # Sampling WITHOUT apply (safe)
    sampled = (
        filtered.groupby("focus_area", group_keys=False)
        .sample(n=rows_per_area, replace=True, random_state=random_state)
    )

    sampled = sampled.reset_index(drop=True)

    sampled.insert(0, "id", ["GOLD{:04d}".format(i + 1) for i in range(len(sampled))])
    sampled["ground_truth"] = sampled["answer"]
    sampled["expected_answer"] = sampled["answer"]

    return sampled


def transform_medquad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["question"] = df["question"].fillna("").map(normalize_text)
    df["answer"] = df["answer"].fillna("").map(normalize_text)
    df["source"] = df["source"].fillna("unknown").map(normalize_text)
    df["focus_area"] = df["focus_area"].fillna("general").map(normalize_text)

    df = df[df["question"].astype(bool) & df["answer"].astype(bool)]
    df["normalized_question"] = df["question"].map(normalize_question)
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
