import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medqa.config import DEFAULT_CONFIG
from medqa.extract import extract_medquad
from medqa.load import save_cleaned, save_golden_dataset, save_url_filtered
from medqa.transform import clean_medquad, enrich_medquad, filter_url_rows, sample_golden_dataset


def main() -> None:
    # --- Extract ---
    print("Loading raw data...")
    raw = extract_medquad(DEFAULT_CONFIG["raw_paths"])
    print(f"  Raw rows: {len(raw)}")

    # --- Step 1: Basic cleaning ---
    print("\nStep 1: Basic cleaning...")
    cleaned = clean_medquad(raw)
    save_cleaned(cleaned)
    print(f"  Rows after cleaning: {len(cleaned)}")
    print(f"  Saved → {DEFAULT_CONFIG['cleaned_csv_path']}")

    # --- Step 2: URL filter ---
    print("\nStep 2: URL filter (keep rows with valid website link in answer)...")
    url_filtered = filter_url_rows(cleaned)
    save_url_filtered(url_filtered)
    print(f"  Rows with valid URL: {len(url_filtered)}")
    print(f"  Saved → {DEFAULT_CONFIG['url_filtered_csv_path']}")

    # --- Step 3: Enrich with question_type and difficulty_level ---
    print("\nStep 3: Enriching with question_type and difficulty_level...")
    enriched = enrich_medquad(url_filtered)
    print(f"  question_type distribution:\n{enriched['question_type'].value_counts().to_string()}")
    print(f"  difficulty_level distribution:\n{enriched['difficulty_level'].value_counts().to_string()}")

    # --- Step 4: Sample golden dataset ---
    print("\nStep 4: Sampling golden dataset (top 10 focus areas, 10 rows each)...")
    golden = sample_golden_dataset(enriched)
    save_golden_dataset(golden)

    saved_cols = DEFAULT_CONFIG["golden_columns"]
    print(f"\n  Total golden records: {len(golden)}")
    print(f"  Saved columns: {saved_cols}")
    print(f"\n  Focus area distribution:")
    for area, count in golden["focus_area"].value_counts().items():
        print(f"    {area}: {count}")
    print(f"\n  question_type distribution per focus_area:")
    print(golden.groupby(["focus_area", "question_type"]).size().to_string())
    print(f"\n  Saved → {DEFAULT_CONFIG['processed_csv_path']}")
    print(f"  Saved → {DEFAULT_CONFIG['processed_path']}")


if __name__ == "__main__":
    main()
