from pathlib import Path
from typing import List

import pandas as pd


def _find_source_file(raw_paths: List[Path]) -> Path:
    for path in raw_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "MedQuAD source file not found. Place medquad.csv in data/raw/ or repository root."
    )


def extract_medquad(raw_paths: List[Path]) -> pd.DataFrame:
    source_path = _find_source_file(raw_paths)
    df = pd.read_csv(source_path, dtype=str)
    if df.empty:
        raise ValueError(f"Loaded MedQuAD file is empty: {source_path}")

    expected_columns = {"question", "answer", "source", "focus_area"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"MedQuAD file missing required columns: {sorted(missing)}")

    return df
