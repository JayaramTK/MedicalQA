"""
Generate Table 2: Baseline vs Basic RAG Comparison.

Reads metrics already computed by run_experiments.py from metrics_tracker.csv
and produces a focused two-row comparison CSV.

Output: data/experiments/baseline_vs_rag_comparison.csv

Usage:
    .venv/bin/python scripts/compare_baseline_vs_rag.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

TRACKER_PATH    = ROOT / "data" / "experiments" / "metrics_tracker.csv"
COMPARISON_PATH = ROOT / "data" / "experiments" / "baseline_vs_rag_comparison.csv"

EVIDENCE_USED = {"E01": "No", "E02": "Yes"}

QUESTION_COUNT = 10   # sample size used in run_experiments.py


# ---------------------------------------------------------------------------
# Key-observation generator
# ---------------------------------------------------------------------------

def _key_observation(row: pd.Series, e01: pd.Series) -> str:
    exp = row["experiment_id"]

    if exp == "E01":
        return (
            f"Baseline LLM relies solely on parametric knowledge with no external retrieval. "
            f"Achieves a faithfulness of {row['faithfulness']:.2f} and an answer relevance of "
            f"{row['answer_relevance']:.2f}, but produces a hallucination rate of "
            f"{row['hallucination_rate']:.2f}, indicating the model occasionally generates "
            f"content not grounded in verified sources."
        )

    if exp == "E02":
        faith_gain  = (row["faithfulness"] - e01["faithfulness"]) / max(e01["faithfulness"], 1e-9)
        hallu_drop  = (e01["hallucination_rate"] - row["hallucination_rate"]) / max(e01["hallucination_rate"], 1e-9)
        rel_gain    = (row["answer_relevance"] - e01["answer_relevance"]) / max(e01["answer_relevance"], 1e-9)
        return (
            f"Basic RAG retrieves relevant context from ChromaDB before answering. "
            f"Faithfulness improves by {faith_gain:.0%} over the baseline "
            f"({e01['faithfulness']:.2f} → {row['faithfulness']:.2f}). "
            f"Hallucination rate drops by {hallu_drop:.0%} "
            f"({e01['hallucination_rate']:.2f} → {row['hallucination_rate']:.2f}), "
            f"and answer relevance gains {rel_gain:.0%}. "
            f"Grounding answers in retrieved evidence substantially improves trustworthiness."
        )

    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not TRACKER_PATH.exists():
        print(f"ERROR: {TRACKER_PATH} not found. Run run_experiments.py first.")
        sys.exit(1)

    tracker = pd.read_csv(TRACKER_PATH)

    missing = [e for e in ("E01", "E02") if e not in tracker["experiment_id"].values]
    if missing:
        print(f"ERROR: Experiments {missing} not found in tracker. Run them first.")
        sys.exit(1)

    e01 = tracker[tracker["experiment_id"] == "E01"].iloc[0]
    e02 = tracker[tracker["experiment_id"] == "E02"].iloc[0]

    rows = []
    for row in (e01, e02):
        rows.append(
            {
                "experiment_id":      row["experiment_id"],
                "system":             row["system_type"],
                "question_count":     QUESTION_COUNT,
                "generated_answers":  QUESTION_COUNT,
                "evidence_used":      EVIDENCE_USED[row["experiment_id"]],
                "faithfulness":       row["faithfulness"],
                "answer_relevance":   row["answer_relevance"],
                "hallucination_rate": row["hallucination_rate"],
                "trust_score":        row["trust_score"],
                "key_observation":    _key_observation(row, e01),
            }
        )

    comparison = pd.DataFrame(rows)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)

    print(f"\nBaseline vs Basic RAG Comparison")
    print("=" * 80)
    pd.set_option("display.max_colwidth", 60)
    print(comparison.drop(columns=["key_observation"]).to_string(index=False))

    print(f"\nKey Observations")
    print("-" * 80)
    for _, r in comparison.iterrows():
        print(f"\n[{r['experiment_id']}] {r['system']}")
        print(f"  {r['key_observation']}")

    print(f"\nSaved → {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
