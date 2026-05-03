from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DEFAULT_CONFIG
from .database import KnowledgeDatabase


def save_golden_dataset(df: pd.DataFrame, processed_path: Path = None) -> Path:
    if processed_path is None:
        processed_path = DEFAULT_CONFIG["processed_path"]

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    return processed_path


def save_csv(df: pd.DataFrame, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return target_path


def build_knowledge_database(df: pd.DataFrame, db_path: Path = None) -> Path:
    if db_path is None:
        db_path = DEFAULT_CONFIG["knowledge_db_path"]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = KnowledgeDatabase(db_path)
    db.create_schema()
    db.insert_records(df.to_dict(orient="records"))
    db.close()
    return db_path
