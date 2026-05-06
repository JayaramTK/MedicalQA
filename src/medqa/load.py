from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG
from .database import KnowledgeDatabase


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_cleaned(df: pd.DataFrame, path: Path = None) -> Path:
    if path is None:
        path = DEFAULT_CONFIG["cleaned_csv_path"]
    _ensure_dir(path)
    df.to_csv(path, index=False)
    return path


def save_url_filtered(df: pd.DataFrame, path: Path = None) -> Path:
    if path is None:
        path = DEFAULT_CONFIG["url_filtered_csv_path"]
    _ensure_dir(path)
    df.to_csv(path, index=False)
    return path


def save_golden_dataset(df: pd.DataFrame, processed_path: Path = None) -> Path:
    if processed_path is None:
        processed_path = DEFAULT_CONFIG["processed_path"]
    csv_path = DEFAULT_CONFIG["processed_csv_path"]

    golden_cols = DEFAULT_CONFIG["golden_columns"]
    output_cols = [c for c in golden_cols if c in df.columns]
    out = df[output_cols]

    _ensure_dir(processed_path)
    out.to_parquet(processed_path, index=False)
    out.to_csv(csv_path, index=False)

    return processed_path


def save_csv(df: pd.DataFrame, target_path: Path) -> Path:
    _ensure_dir(target_path)
    df.to_csv(target_path, index=False)
    return target_path


def build_knowledge_database(df: pd.DataFrame, db_path: Path = None) -> Path:
    if db_path is None:
        db_path = DEFAULT_CONFIG["knowledge_db_path"]

    _ensure_dir(db_path)
    db = KnowledgeDatabase(db_path)
    db.create_schema()
    db.insert_records(df.to_dict(orient="records"))
    db.close()
    return db_path
