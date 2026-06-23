"""
Build the golden evaluation dataset.

Samples GOLDEN_TARGET_ROWS unique-question rows from the top focus areas,
maximising question_type diversity within each area (rarest type first).
"""
import pandas as pd

from .config import (
    GOLDEN_COLUMNS, GOLDEN_CSV_PATH, GOLDEN_PARQUET_PATH,
    GOLDEN_RANDOM_STATE, GOLDEN_ROWS_PER_AREA, GOLDEN_TARGET_ROWS,
)


def sample_golden(
    df: pd.DataFrame,
    target_rows: int = GOLDEN_TARGET_ROWS,
    rows_per_area: int = GOLDEN_ROWS_PER_AREA,
    random_state: int = GOLDEN_RANDOM_STATE,
) -> pd.DataFrame:
    areas_in_order = df["focus_area"].value_counts().index.tolist()
    parts: list[pd.DataFrame] = []
    collected = 0

    for area in areas_in_order:
        if collected >= target_rows:
            break
        area_df = df[df["focus_area"] == area].copy()
        need    = min(len(area_df), rows_per_area, target_rows - collected)

        # Diversity pass — one row per question_type, rarest type first
        type_order   = area_df["question_type"].value_counts().index.tolist()[::-1]
        selected_idx: list = []
        for qtype in type_order:
            candidates = area_df[
                (area_df["question_type"] == qtype) & (~area_df.index.isin(selected_idx))
            ]
            if not candidates.empty:
                selected_idx.append(candidates.sample(1, random_state=random_state).index[0])
            if len(selected_idx) >= need:
                break

        # Fill pass — no replacement
        if len(selected_idx) < need:
            remaining = area_df[~area_df.index.isin(selected_idx)]
            extra     = remaining.sample(min(need - len(selected_idx), len(remaining)),
                                         random_state=random_state)
            selected_idx.extend(extra.index.tolist())

        parts.append(area_df.loc[selected_idx[:need]])
        collected += need

    sampled = pd.concat(parts).reset_index(drop=True)
    sampled.insert(0, "golden_id", [f"GOLD{i+1:04d}" for i in range(len(sampled))])
    sampled["ground_truth"]     = sampled["answer"]
    sampled["expected_context"] = sampled["answer"]

    print(f"  [GoldenDataset] {len(sampled):,} rows | "
          f"{sampled['question'].nunique()} unique questions | "
          f"{sampled['focus_area'].nunique()} focus areas")
    return sampled


def save(golden: pd.DataFrame) -> None:
    output_cols = [c for c in GOLDEN_COLUMNS if c in golden.columns]
    out = golden[output_cols]
    for path in (GOLDEN_CSV_PATH, GOLDEN_PARQUET_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GOLDEN_CSV_PATH, index=False)
    out.to_parquet(GOLDEN_PARQUET_PATH, index=False)
    print(f"  [GoldenDataset] Saved → {GOLDEN_CSV_PATH}")
    print(f"  [GoldenDataset] Saved → {GOLDEN_PARQUET_PATH}")


if __name__ == "__main__":
    from .extract import extract
    from .clean import clean
    from .enrich import enrich
    save(sample_golden(enrich(clean(extract()))))
