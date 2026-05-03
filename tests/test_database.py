import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medqa.database import KnowledgeDatabase


def test_database_inserts_records(tmp_path: Path) -> None:
    db_path = tmp_path / "medqa_test.db"
    db = KnowledgeDatabase(db_path)
    db.create_schema()

    records = [
        {"question": "What is diabetes?", "answer": "A metabolic disorder.", "source": "NIH", "focus_area": "Endocrine"}
    ]
    db.insert_records(records)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT count(*) FROM documents")
        assert cursor.fetchone()[0] == 1

    results = db.query("What is diabetes?")
    assert results
    assert results[0]["question"] == "What is diabetes?"
    db.close()
