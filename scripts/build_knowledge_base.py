import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medqa.config import DEFAULT_CONFIG
from medqa.knowledge import build_knowledge_base
from medqa.load import save_knowledge_base


def main() -> None:
    # --- Load golden dataset ---
    golden_path = DEFAULT_CONFIG["processed_csv_path"]
    print(f"Loading golden dataset from: {golden_path}")
    golden = pd.read_csv(golden_path)
    print(f"  Rows loaded: {len(golden)}")

    # --- Build knowledge base ---
    print("\nBuilding knowledge base...")
    kb = build_knowledge_base(golden)

    # --- Save ---
    csv_path, parquet_path = save_knowledge_base(kb)

    # --- Report ---
    print(f"\n  Documents in knowledge base : {len(kb)}")
    print(f"  Columns                     : {list(kb.columns)}")
    print(f"  Null values per column:")
    for col in kb.columns:
        n = kb[col].isna().sum() + kb[col].eq('').sum()
        print(f"    {col:<20} {n} nulls/empty")

    print(f"\n  medical_topic distribution:")
    for topic, count in kb['medical_topic'].value_counts().items():
        print(f"    {topic:<50} {count}")

    print(f"\n  Saved → {csv_path}")
    print(f"  Saved → {parquet_path}")

    print("\n  Sample rows:")
    pd.set_option('display.max_colwidth', 60)
    print(kb[['document_id', 'title', 'source', 'medical_topic', 'source_link']].head(5).to_string())


if __name__ == "__main__":
    main()
