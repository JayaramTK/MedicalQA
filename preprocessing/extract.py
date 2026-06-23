"""Extract raw MedQuAD data — locate the CSV, validate schema, return a DataFrame."""
from pathlib import Path

import pandas as pd

from .config import RAW_PATHS


def find_source_file(raw_paths: list[Path] = RAW_PATHS) -> Path:
    for path in raw_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "MedQuAD source file not found. Place medquad.csv in data/raw/ or the repository root."
    )


def extract(raw_paths: list[Path] = RAW_PATHS) -> pd.DataFrame:
    source = find_source_file(raw_paths)
    df = pd.read_csv(source, dtype=str)

    if df.empty:
        raise ValueError(f"Loaded file is empty: {source}")

    required = {"question", "answer", "source", "focus_area"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"  [Extract] {len(df):,} rows loaded from {source.name}")
    return df


if __name__ == "__main__":
    print(extract().head(3).to_string())
