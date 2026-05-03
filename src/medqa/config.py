from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = {
    "raw_paths": [REPO_ROOT / "data" / "raw" / "medquad.csv", REPO_ROOT / "medquad.csv"],
    "processed_path": REPO_ROOT / "data" / "processed" / "golden_medquad.parquet",
    "knowledge_db_path": REPO_ROOT / "data" / "knowledge" / "medquad_knowledge.db",
    "golden_columns": ["question", "answer", "source", "focus_area", "normalized_question"],
}
