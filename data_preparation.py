"""
Root orchestrator CLI — runs the full data preparation pipeline.

Usage:
    python data_preparation.py                   # run all stages
    python data_preparation.py --stages extract clean enrich
    python data_preparation.py --stages golden_dataset
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_extract(data=None):
    from preprocessing.extract import extract
    return extract()


def run_clean(data=None):
    from preprocessing.extract import extract
    from preprocessing.clean import clean, save
    df = data if data is not None else extract()
    cleaned = clean(df)
    save(cleaned)
    return cleaned


def run_enrich(data=None):
    from preprocessing.extract import extract
    from preprocessing.clean import clean
    from preprocessing.enrich import enrich, save_url_filtered
    df = data if data is not None else clean(extract())
    enriched = enrich(df)
    save_url_filtered(enriched)
    return enriched


def run_knowledge_base(data=None):
    from preprocessing.extract import extract
    from preprocessing.clean import clean
    from preprocessing.enrich import enrich
    from preprocessing.knowledge_base import build_knowledge_base, save
    # Expects enriched raw data (has 'focus_area'); re-run if KB data passed in
    df = data if (data is not None and "focus_area" in data.columns) \
         else enrich(clean(extract()))
    kb = build_knowledge_base(df)
    save(kb)
    return df   # return enriched data so golden_dataset can reuse it


def run_golden_dataset(data=None):
    from preprocessing.extract import extract
    from preprocessing.clean import clean
    from preprocessing.enrich import enrich
    from preprocessing.golden_dataset import sample_golden, save
    # Expects enriched raw data (has 'focus_area')
    df = data if (data is not None and "focus_area" in data.columns) \
         else enrich(clean(extract()))
    golden = sample_golden(df)
    save(golden)
    return golden


STAGES = {
    "extract":        run_extract,
    "clean":          run_clean,
    "enrich":         run_enrich,
    "knowledge_base": run_knowledge_base,
    "golden_dataset": run_golden_dataset,
}

STAGE_ORDER = ["extract", "clean", "enrich", "knowledge_base", "golden_dataset"]

DESCRIPTIONS = {
    "extract":        "Load and validate raw MedQuAD CSV",
    "clean":          "Remove noise, deduplicate, reset index",
    "enrich":         "Resolve website links, classify question type & difficulty",
    "knowledge_base": "Build cleaned knowledge base (CSV + Parquet)",
    "golden_dataset": "Sample 100-row golden evaluation dataset",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MedQA Data Preparation Pipeline")
    parser.add_argument(
        "--stages", nargs="+", choices=STAGE_ORDER,
        help="Stages to run in order (default: all)",
    )
    args = parser.parse_args()

    selected = args.stages or STAGE_ORDER

    print("\nMedQA Data Preparation Pipeline")
    print("=" * 50)
    for stage in selected:
        print(f"  {stage:<20}  {DESCRIPTIONS[stage]}")
    print("=" * 50)

    data = None
    for stage in selected:
        print(f"\n── {DESCRIPTIONS[stage]} ──")
        data = STAGES[stage](data)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
