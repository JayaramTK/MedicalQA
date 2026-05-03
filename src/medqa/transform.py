from typing import Tuple

import pandas as pd

from .text import build_dedup_key, normalize_question, normalize_text


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
