from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DATA_KNOWLEDGE_DIR = REPO_ROOT / "data" / "knowledge"

DEFAULT_CONFIG = {
    "raw_paths": [DATA_RAW_DIR / "medquad.csv", REPO_ROOT / "medquad.csv"],
    "cleaned_csv_path": DATA_PROCESSED_DIR / "medquad_cleaned.csv",
    "url_filtered_csv_path": DATA_PROCESSED_DIR / "medquad_url_filtered.csv",
    "processed_path": DATA_PROCESSED_DIR / "golden_medquad.parquet",
    "processed_csv_path": DATA_PROCESSED_DIR / "golden_medquad.csv",
    "knowledge_db_path": DATA_KNOWLEDGE_DIR / "medquad_knowledge.db",
    "golden_columns": [
        "golden_id",
        "question",
        "ground_truth",
        "source",
        "focus_area",
        "expected_context",
        "context_source_id",
        "question_type",
        "difficulty_level",
    ],
}
