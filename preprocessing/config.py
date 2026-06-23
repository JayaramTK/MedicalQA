"""
Global configuration — paths, seeds, and constants shared across all pipeline steps.
"""
from pathlib import Path

# Repository root (two levels up from this file)
ROOT = Path(__file__).resolve().parent.parent

# ── Data directories ────────────────────────────────────────────────────────
DATA_RAW_DIR           = ROOT / "data" / "raw"
DATA_PROCESSED_DIR     = ROOT / "data" / "processed"
DATA_KNOWLEDGE_BASE_DIR = ROOT / "data" / "knowledge_base"
DATA_GOLDEN_DIR        = ROOT / "data" / "golden_dataset"

# ── Raw inputs ──────────────────────────────────────────────────────────────
RAW_PATHS = [
    DATA_RAW_DIR / "medquad.csv",
]

# ── Intermediate processed files ────────────────────────────────────────────
CLEANED_CSV_PATH      = DATA_PROCESSED_DIR / "medquad_cleaned.csv"
URL_FILTERED_CSV_PATH = DATA_PROCESSED_DIR / "medquad_url_filtered.csv"

# ── Knowledge base outputs ──────────────────────────────────────────────────
KNOWLEDGE_BASE_CSV     = DATA_KNOWLEDGE_BASE_DIR / "knowledge_base.csv"
KNOWLEDGE_BASE_PARQUET = DATA_KNOWLEDGE_BASE_DIR / "knowledge_base.parquet"

# ── Golden dataset outputs ──────────────────────────────────────────────────
GOLDEN_CSV_PATH     = DATA_GOLDEN_DIR / "golden_medquad.csv"
GOLDEN_PARQUET_PATH = DATA_GOLDEN_DIR / "golden_medquad.parquet"

# ── Results directory ───────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "results"

# ── Cleaning thresholds ─────────────────────────────────────────────────────
MIN_ANSWER_WORDS   = 100          # answers shorter than this are dropped
MIN_ALPHA_RATIO    = 0.70         # answers with < 70% alphabetic chars are dropped

# ── Golden dataset sampling ─────────────────────────────────────────────────
GOLDEN_TARGET_ROWS   = 100
GOLDEN_ROWS_PER_AREA = 10
GOLDEN_RANDOM_STATE  = 42

# ── Source-institution → website URL mapping ────────────────────────────────
SOURCE_URL_MAP: dict[str, str] = {
    "GHR":              "https://ghr.nlm.nih.gov",
    "GARD":             "https://rarediseases.info.nih.gov",
    "NIDDK":            "https://www.niddk.nih.gov",
    "NINDS":            "https://www.ninds.nih.gov",
    "MPlusHealthTopics":"https://medlineplus.gov",
    "NIHSeniorHealth":  "https://nihseniorhealth.gov",
    "CancerGov":        "https://www.cancer.gov",
    "NHLBI":            "https://www.nhlbi.nih.gov",
    "CDC":              "https://www.cdc.gov",
}

# ── Golden dataset output columns ───────────────────────────────────────────
GOLDEN_COLUMNS = [
    "golden_id",
    "question",
    "ground_truth",
    "source",
    "focus_area",
    "expected_context",
    "context_source_id",
    "question_type",
    "difficulty_level",
]
