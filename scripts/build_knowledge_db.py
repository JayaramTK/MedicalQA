import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medqa.config import DEFAULT_CONFIG
from medqa.database import KnowledgeDatabase
from medqa.extract import extract_medquad
from medqa.transform import transform_medquad


def main() -> None:
    df = extract_medquad(DEFAULT_CONFIG["raw_paths"])
    processed = transform_medquad(df)
    db = KnowledgeDatabase(DEFAULT_CONFIG["knowledge_db_path"])
    db.create_schema()
    db.insert_records(processed.to_dict(orient="records"))
    db.close()
    print(f"Knowledge database built at: {DEFAULT_CONFIG['knowledge_db_path']}")
    print(f"Indexed records: {len(processed)}")


if __name__ == "__main__":
    main()
