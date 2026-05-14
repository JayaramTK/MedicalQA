"""
Run all five experiments (E01-E05) and update data/experiments/metrics_tracker.csv.

Usage:
    .venv/bin/python scripts/run_experiments.py                    # all 5, 10 questions
    .venv/bin/python scripts/run_experiments.py --sample 20        # custom sample size
    .venv/bin/python scripts/run_experiments.py --all              # full golden dataset
    .venv/bin/python scripts/run_experiments.py --experiments E05  # re-run E05 only
    .venv/bin/python scripts/run_experiments.py --experiments E04 E05  # re-run E04 & E05
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medqa.config import DEFAULT_CONFIG
from medqa.experiments import (
    E01_BaselineLLM,
    E02_BasicRAG,
    E03_SentenceEvidence,
    E04_UncertaintyRAG,
    E05_CombinedRAG,
)

TRACKER_PATH = ROOT / "data" / "experiments" / "metrics_tracker.csv"

METRIC_COLS = [
    "faithfulness",
    "context_relevance",
    "context_precision",
    "answer_relevance",
    "hallucination_rate",
    "trust_score",
]

ALL_EXPERIMENTS = {
    "E01": E01_BaselineLLM,
    "E02": E02_BasicRAG,
    "E03": E03_SentenceEvidence,
    "E04": E04_UncertaintyRAG,
    "E05": E05_CombinedRAG,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_golden(sample: int | None) -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_CONFIG["processed_csv_path"])
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    return df


def load_knowledge_base() -> list[dict]:
    kb_path = DEFAULT_CONFIG["knowledge_db_path"].parent / "knowledge_base.csv"
    return pd.read_csv(kb_path).to_dict(orient="records")


def update_tracker(exp_config: dict, avg_metrics: dict[str, float]) -> None:
    """Write / overwrite the row for this experiment in the tracker CSV."""
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {**exp_config, **avg_metrics}

    if TRACKER_PATH.exists():
        tracker = pd.read_csv(TRACKER_PATH)
        mask = tracker["experiment_id"] == exp_config["experiment_id"]
        if mask.any():
            for col, val in row.items():
                tracker.loc[mask, col] = val
        else:
            tracker = pd.concat(
                [tracker, pd.DataFrame([row])], ignore_index=True
            )
    else:
        tracker = pd.DataFrame([row])

    tracker.to_csv(TRACKER_PATH, index=False)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp_class,
    questions: pd.DataFrame,
    knowledge_base: list[dict],
) -> dict[str, float]:
    exp = exp_class(knowledge_base)
    cfg = exp_class.CONFIG
    print(f"\n{'='*60}")
    print(f"  {cfg['experiment_id']} — {cfg['system_type']}")
    print(f"{'='*60}")

    exp.setup()
    per_q_metrics: list[dict] = []

    for i, row in questions.iterrows():
        q  = row["question"]
        gt = row["ground_truth"]
        print(f"  [{i+1:02d}/{len(questions)}] {q[:70]}...")

        result = exp.run_question(q, gt)
        per_q_metrics.append(result.metrics)

        for k, v in result.metrics.items():
            print(f"       {k:<22} {v:.4f}")

    # Average metrics across all questions
    avg: dict[str, float] = {}
    for col in METRIC_COLS:
        vals = [m[col] for m in per_q_metrics]
        avg[col] = round(sum(vals) / len(vals), 4)

    print(f"\n  ── Averages for {cfg['experiment_id']} ──")
    for k, v in avg.items():
        print(f"     {k:<22} {v:.4f}")

    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--sample", type=int, default=10,
                            help="Number of questions to sample (default: 10)")
    size_group.add_argument("--all", action="store_true",
                            help="Run on the full golden dataset")
    parser.add_argument(
        "--experiments", nargs="+", metavar="ID",
        choices=list(ALL_EXPERIMENTS.keys()),
        help="Run only specific experiments, e.g. --experiments E05",
    )
    args = parser.parse_args()

    selected = (
        [ALL_EXPERIMENTS[e] for e in args.experiments]
        if args.experiments
        else list(ALL_EXPERIMENTS.values())
    )

    sample_size = None if args.all else args.sample
    questions   = load_golden(sample_size)
    kb_docs     = load_knowledge_base()

    print(f"\nRunning {len(selected)} experiment(s) on {len(questions)} questions.")
    print(f"Knowledge base: {len(kb_docs)} documents\n")

    for exp_class in selected:
        avg = run_experiment(exp_class, questions, kb_docs)
        update_tracker(exp_class.CONFIG, avg)
        print(f"  → Tracker updated: {TRACKER_PATH}")

    print("\n\n── Final Metrics Tracker ──")
    print(pd.read_csv(TRACKER_PATH).to_string(index=False))


if __name__ == "__main__":
    main()
