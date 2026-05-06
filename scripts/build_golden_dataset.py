import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medqa.config import DEFAULT_CONFIG
from medqa.extract import extract_medquad
from medqa.load import save_cleaned, save_golden_dataset, save_url_filtered
from medqa.transform import (
    clean_medquad,
    enrich_medquad,
    filter_url_rows,
    resolve_website_links,
    sample_golden_dataset,
)


def main() -> None:
    # --- Extract ---
    print("Loading raw data...")
    raw = extract_medquad(DEFAULT_CONFIG["raw_paths"])
    print(f"  Raw rows: {len(raw)}")

    # --- Step 1: Basic cleaning ---
    print("\nStep 1: Basic cleaning...")
    cleaned = clean_medquad(raw)
    save_cleaned(cleaned)
    print(f"  Rows after cleaning : {len(cleaned)}")
    print(f"  Saved → {DEFAULT_CONFIG['cleaned_csv_path']}")

    # --- Step 2a: Literal URL filter (intermediate reference file) ---
    print("\nStep 2a: Literal URL filter (rows with http URL directly in answer)...")
    url_literal = filter_url_rows(cleaned)
    save_url_filtered(url_literal)
    print(f"  Rows with literal URL in answer : {len(url_literal)}")
    print(f"  Saved → {DEFAULT_CONFIG['url_filtered_csv_path']}")

    # --- Step 2b: Resolve website link for every row ---
    print("\nStep 2b: Resolving website link for all rows "
          "(answer URL → source institution URL)...")
    linked = resolve_website_links(cleaned)
    print(f"  Rows with resolved website link : {len(linked)}")
    print(f"  context_source_id null count    : {linked['context_source_id'].eq('').sum()}")

    # --- Step 3: Enrich with question_type and difficulty_level ---
    print("\nStep 3: Enriching with question_type and difficulty_level...")
    enriched = enrich_medquad(linked)
    print(f"  question_type  : {enriched['question_type'].value_counts().to_dict()}")
    print(f"  difficulty_level: {enriched['difficulty_level'].value_counts().to_dict()}")

    # --- Step 4: Sample golden dataset ---
    print("\nStep 4: Sampling golden dataset (100 unique rows, ≤10 per focus area)...")
    golden = sample_golden_dataset(enriched)
    save_golden_dataset(golden)

    print(f"\n  Total golden records  : {len(golden)}")
    print(f"  Unique questions      : {golden['question'].nunique()}")
    print(f"  context_source_id populated : "
          f"{golden['context_source_id'].replace('', None).notna().sum()} / {len(golden)}")

    print(f"\n  Focus area breakdown (rows | q-types):")
    for area, grp in golden.groupby("focus_area"):
        types = grp["question_type"].value_counts().to_dict()
        src   = grp["context_source_id"].iloc[0]
        print(f"    {area[:48]:<48}  {len(grp):>2} rows | {types}")
        print(f"    {'':48}  source → {src}")

    print(f"\n  Saved → {DEFAULT_CONFIG['processed_csv_path']}")
    print(f"  Saved → {DEFAULT_CONFIG['processed_path']}")


if __name__ == "__main__":
    main()
