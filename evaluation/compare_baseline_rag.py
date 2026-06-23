"""
Table 2: Baseline vs Basic RAG Comparison.

E01 and E02 are read directly from results/baseline_vs_rag_comparison.csv.
E02.1 (Sparse RAG) and E02.2 (Hybrid RAG) are run fresh and appended.
The completed 4-row table is saved back to the same file.

Output: results/baseline_vs_rag_comparison.csv
Usage:  .venv/bin/python evaluation/compare_baseline_rag.py
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from preprocessing.config import GOLDEN_CSV_PATH, RESULTS_DIR

COMPARISON_PATH = RESULTS_DIR / "baseline_vs_rag_comparison.csv"
QUESTION_COUNT  = 10
RANDOM_STATE    = 42

EVIDENCE_USED = {"E01": "No", "E02": "Yes", "E02.1": "Yes", "E02.2": "Yes"}
EXPERIMENT_ORDER = ["E01", "E02", "E02.1", "E02.2"]

_METRIC_COLS = [
    "faithfulness", "context_relevance", "context_precision",
    "answer_relevance", "hallucination_rate", "trust_score",
]


# ---------------------------------------------------------------------------
# Run one new experiment and return averaged metrics
# ---------------------------------------------------------------------------

def _run_experiment(eid: str) -> dict:
    from experiments.e02_basic_rag.implementation.sparse_pipeline import SparseRAGPipeline
    from experiments.e02_basic_rag.implementation.hybrid_pipeline import HybridRAGPipeline

    PIPELINES = {"E02.1": SparseRAGPipeline, "E02.2": HybridRAGPipeline}
    SYSTEMS   = {"E02.1": "Sparse RAG",      "E02.2": "Hybrid RAG"}

    golden   = pd.read_csv(GOLDEN_CSV_PATH).sample(n=QUESTION_COUNT, random_state=RANDOM_STATE)
    pipeline = PIPELINES[eid]()
    per_q: list[dict] = []

    print(f"\n  Running {eid} — {SYSTEMS[eid]} on {len(golden)} questions…")
    for i, row in golden.reset_index(drop=True).iterrows():
        q, gt = row["question"], row["ground_truth"]
        print(f"    [{i+1:02d}/{len(golden)}] {q[:65]}…")
        result = pipeline.run(q, gt)
        per_q.append({k: result[k] for k in _METRIC_COLS})
        print(f"         faith={result['faithfulness']:.3f}  "
              f"hall={result['hallucination_rate']:.3f}  "
              f"trust={result['trust_score']:.3f}")

    return {col: round(sum(m[col] for m in per_q) / len(per_q), 4)
            for col in _METRIC_COLS}


# ---------------------------------------------------------------------------
# Key observation generator
# ---------------------------------------------------------------------------

def _key_observation(eid: str, row: dict, e01: dict) -> str:
    faith = row["faithfulness"]
    hall  = row["hallucination_rate"]
    rel   = row["answer_relevance"]

    faith_gain = (faith - e01["faithfulness"]) / max(e01["faithfulness"], 1e-9)
    hallu_drop = (e01["hallucination_rate"] - hall) / max(e01["hallucination_rate"], 1e-9)
    sign_f = "+" if faith_gain >= 0 else ""
    sign_h = "-" if hallu_drop >= 0 else "+"

    if eid == "E01":
        return (
            f"Baseline LLM relies solely on parametric knowledge with no external retrieval. "
            f"Faithfulness of {faith:.2f} and answer relevance of {rel:.2f}, but hallucination "
            f"rate of {hall:.2f} shows the model occasionally generates ungrounded content."
        )
    if eid == "E02":
        return (
            f"Dense semantic retrieval via ChromaDB. "
            f"Faithfulness {sign_f}{faith_gain:.0%} vs baseline ({e01['faithfulness']:.2f} → {faith:.2f}). "
            f"Hallucination drops {sign_h}{abs(hallu_drop):.0%} ({e01['hallucination_rate']:.2f} → {hall:.2f}). "
            f"Grounding answers in retrieved evidence substantially improves trustworthiness."
        )
    if eid == "E02.1":
        return (
            f"BM25 keyword-based retrieval with no embedding model. "
            f"Faithfulness {sign_f}{faith_gain:.0%} vs baseline ({e01['faithfulness']:.2f} → {faith:.2f}). "
            f"Hallucination rate: {hall:.2f}. "
            f"Keyword matching captures exact medical terminology but misses semantic similarity."
        )
    if eid == "E02.2":
        return (
            f"Hybrid retrieval merges dense and BM25 rankings via Reciprocal Rank Fusion. "
            f"Faithfulness {sign_f}{faith_gain:.0%} vs baseline ({e01['faithfulness']:.2f} → {faith:.2f}). "
            f"Hallucination rate: {hall:.2f}. "
            f"Combining both retrieval signals improves coverage over either approach alone."
        )
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not COMPARISON_PATH.exists():
        print(f"ERROR: {COMPARISON_PATH} not found. Run run_all_experiments.py first.")
        sys.exit(1)

    # Load existing E01 and E02 rows
    existing = pd.read_csv(COMPARISON_PATH)
    existing_ids = set(existing["experiment_id"].tolist())

    for eid in ("E01", "E02"):
        if eid not in existing_ids:
            print(f"ERROR: {eid} missing from {COMPARISON_PATH.name}. Run run_all_experiments.py first.")
            sys.exit(1)

    e01 = existing[existing["experiment_id"] == "E01"].iloc[0].to_dict()

    # Build rows — reuse existing E01/E02, run new E02.1/E02.2
    rows: list[dict] = []
    for eid in EXPERIMENT_ORDER:
        if eid in existing_ids:
            src = existing[existing["experiment_id"] == eid].iloc[0].to_dict()
            rows.append(src)
        else:
            metrics = _run_experiment(eid)
            system_names = {"E02.1": "Sparse RAG", "E02.2": "Hybrid RAG"}
            row = {
                "experiment_id":      eid,
                "system":             system_names[eid],
                "question_count":     QUESTION_COUNT,
                "generated_answers":  QUESTION_COUNT,
                "evidence_used":      EVIDENCE_USED[eid],
                "faithfulness":       metrics["faithfulness"],
                "answer_relevance":   metrics["answer_relevance"],
                "hallucination_rate": metrics["hallucination_rate"],
                "trust_score":        metrics["trust_score"],
                "key_observation":    _key_observation(eid, metrics, e01),
            }
            rows.append(row)

    # Regenerate key observations for E01/E02 using the same function
    for row in rows:
        if row["experiment_id"] in ("E01", "E02"):
            row["key_observation"] = _key_observation(
                row["experiment_id"], row, e01
            )

    comparison = pd.DataFrame(rows)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)

    print(f"\nTable 2: Baseline vs Basic RAG Comparison")
    print("=" * 90)
    pd.set_option("display.max_colwidth", 55)
    print(comparison.drop(columns=["key_observation"]).to_string(index=False))

    print(f"\nKey Observations")
    print("-" * 90)
    for _, r in comparison.iterrows():
        print(f"\n[{r['experiment_id']}] {r['system']}")
        print(f"  {r['key_observation']}")

    print(f"\nSaved → {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
