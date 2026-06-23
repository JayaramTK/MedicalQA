"""
Table 4: Uncertainty-Aware RAG Novelty Result.

One row per threshold value. The system refuses to answer when the max
retrieval score falls below the threshold.

All thresholds: [0, 0.2, 0.3, 0.5, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9]

Already-computed rows are loaded from the output CSV and skipped —
only missing thresholds are run.

Metrics:
  answered_questions  — questions where max_score >= threshold
  refused_questions   — questions where max_score <  threshold
  correct_refusals    — refused AND max_score < CORRECT_REFUSAL_CUTOFF
  wrong_refusals      — refused AND max_score >= CORRECT_REFUSAL_CUTOFF
  refusal_accuracy    — correct_refusals / max(refused, 1)
  hallucination_rate  — averaged over answered questions only
  trust_score         — 0.4*(1-hall) + 0.3*refusal_accuracy + 0.3*(answered/total)

Output: results/uncertainty_rag_analysis.csv
Usage : .venv/bin/python evaluation/uncertainty_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from preprocessing.config import GOLDEN_CSV_PATH, RESULTS_DIR
from shared.embedder   import Embedder
from shared.llm_client import LLMClient
from shared.metrics    import compute_metrics
from experiments.e04_uncertainty_rag.implementation.ingestion import build_vector_store
from experiments.e04_uncertainty_rag.implementation.config    import TOP_K

OUTPUT_PATH            = RESULTS_DIR / "uncertainty_rag_analysis.csv"
QUESTION_COUNT         = 10
RANDOM_STATE           = 42
CORRECT_REFUSAL_CUTOFF = 0.84   # refusal is "correct" when max_score < this
ALL_THRESHOLDS         = [0.0, 0.2, 0.3, 0.5, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9]


# ---------------------------------------------------------------------------
# Load previously completed rows (from either output file)
# ---------------------------------------------------------------------------

def _is_complete(row: dict) -> bool:
    val = row.get("hallucination_rate", None)
    try:
        return val is not None and str(val).strip() not in ("", "nan") and float(val) >= 0
    except (ValueError, TypeError):
        return False


def load_existing() -> dict[float, dict]:
    """Return complete rows keyed by threshold from the output CSV."""
    if not OUTPUT_PATH.exists():
        return {}
    complete: dict[float, dict] = {}
    for _, row in pd.read_csv(OUTPUT_PATH).iterrows():
        r = row.to_dict()
        if _is_complete(r):
            complete[round(float(r["threshold"]), 10)] = r
    return complete


# ---------------------------------------------------------------------------
# Build one result row for a given threshold
# ---------------------------------------------------------------------------

def _build_row(
    threshold: float,
    total: int,
    answered_metrics: list[dict],
    refused_scores: list[float],
) -> dict:
    answered = len(answered_metrics)
    refused  = len(refused_scores)

    correct_refusals = sum(1 for s in refused_scores if s < CORRECT_REFUSAL_CUTOFF)
    wrong_refusals   = refused - correct_refusals
    refusal_accuracy = round(correct_refusals / max(refused, 1), 4)

    hallucination_rate = (
        round(sum(m["hallucination_rate"] for m in answered_metrics) / answered, 4)
        if answered_metrics else 1.0
    )
    trust_score = round(
        0.4 * (1 - hallucination_rate)
        + 0.3 * refusal_accuracy
        + 0.3 * (answered / total),
        4,
    )

    print(f"  → threshold={threshold}  answered={answered}  refused={refused}  "
          f"hall={hallucination_rate:.3f}  trust={trust_score:.3f}")

    return {
        "experiment_id":      "E04",
        "system":             "RAG with Uncertainty Handling",
        "threshold":          threshold,
        "total_questions":    total,
        "answered_questions": answered,
        "refused_questions":  refused,
        "correct_refusals":   correct_refusals,
        "wrong_refusals":     wrong_refusals,
        "hallucination_rate": hallucination_rate,
        "refusal_accuracy":   refusal_accuracy,
        "trust_score":        trust_score,
    }


# ---------------------------------------------------------------------------
# Run one threshold value
# ---------------------------------------------------------------------------

def run_threshold(
    threshold: float,
    golden: pd.DataFrame,
    vs,
    embedder: Embedder,
    llm: LLMClient,
    split_cache: dict,
) -> dict:
    """Run E04 for one threshold.

    Retrieval scores are pre-computed to determine the refused set.
    If a previous threshold produced the identical refused set, the LLM
    metrics are reused from split_cache (no extra API calls).
    """
    total = len(golden)

    # Retrieval pass — no LLM calls
    max_scores: dict[int, float] = {}
    for i, row in golden.iterrows():
        hits = vs.query(row["question"], n_results=TOP_K)
        max_scores[i] = max(h["score"] for h in hits) if hits else 0.0

    refused_indices = frozenset(i for i, s in max_scores.items() if s < threshold)
    refused_scores  = [max_scores[i] for i in sorted(refused_indices)]

    # Cache hit — same refused/answered split
    if refused_indices in split_cache:
        print(f"\n  E04 threshold={threshold} — same split as previous run, reusing metrics.")
        return _build_row(threshold, total, split_cache[refused_indices], refused_scores)

    # New split — run LLM for answered questions
    print(f"\n  Running E04 (threshold={threshold}) on {total} questions…")
    answered_metrics: list[dict] = []

    for i, row in golden.iterrows():
        q, gt = row["question"], row["ground_truth"]
        if i in refused_indices:
            print(f"    [{i+1:02d}/{total}] REFUSED  max_score={max_scores[i]:.4f} < {threshold}")
            continue

        hits   = vs.query(q, n_results=TOP_K)
        chunks = [h["text"] for h in hits]
        scores = [h["score"] for h in hits]
        conf   = "High" if max_scores[i] >= 0.84 else "Medium"
        answer = llm.answer_uncertainty(q, "\n\n".join(chunks), conf)

        metrics = compute_metrics(
            question=q, answer=answer, ground_truth=gt,
            retrieved_chunks=chunks, retrieval_scores=scores,
            embedder=embedder, llm=llm, has_rag=True,
        )
        answered_metrics.append(metrics)
        print(f"    [{i+1:02d}/{total}] Answered  max_score={max_scores[i]:.4f}  "
              f"faith={metrics['faithfulness']:.3f}  "
              f"hall={metrics['hallucination_rate']:.3f}")

    split_cache[refused_indices] = answered_metrics
    return _build_row(threshold, total, answered_metrics, refused_scores)


# ---------------------------------------------------------------------------
# Checkpoint save after each threshold
# ---------------------------------------------------------------------------

def _save_checkpoint(new_row: dict, all_existing: dict[float, dict]) -> None:
    all_existing[round(float(new_row["threshold"]), 10)] = new_row
    ordered = [all_existing[t] for t in sorted(all_existing)]
    _attach_observations(ordered)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ordered).to_csv(OUTPUT_PATH, index=False)
    print(f"  → Checkpoint saved → {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Key observation
# ---------------------------------------------------------------------------

def _key_observation(row: dict, all_rows: list[dict]) -> str:
    thr   = row["threshold"]
    ref   = row["refused_questions"]
    total = row["total_questions"]
    hall  = row["hallucination_rate"]
    acc   = row["refusal_accuracy"]

    if ref == 0:
        return (
            f"At threshold={thr}, all {total} questions are answered "
            f"(retrieval scores exceed threshold). "
            f"Hallucination rate: {hall:.2f}. No refusals triggered."
        )

    max_thr = max(r["threshold"] for r in all_rows)
    coverage = round(row["answered_questions"] / total, 2)
    cr_pct   = f"{row['correct_refusals']}/{ref}"
    tail = (
        "Stricter threshold reduces hallucination but limits coverage."
        if thr == max_thr
        else "Balances coverage and confidence."
    )
    return (
        f"Threshold={thr} refuses {ref}/{total} questions "
        f"({ref/total:.0%} refusal rate). "
        f"Coverage: {coverage:.0%}. "
        f"Correct refusals: {cr_pct} (accuracy={acc:.2f}). "
        f"Hallucination rate on answered: {hall:.2f}. {tail}"
    )


def _attach_observations(rows: list[dict]) -> None:
    for r in rows:
        r["key_observation"] = _key_observation(r, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    existing   = load_existing()
    to_run     = [t for t in ALL_THRESHOLDS if t not in existing]
    skipped    = [t for t in ALL_THRESHOLDS if t in existing]

    if skipped:
        print(f"\nSkipping already-completed thresholds: {skipped}")
    if to_run:
        print(f"Will run thresholds: {to_run}")

    if to_run:
        golden   = pd.read_csv(GOLDEN_CSV_PATH).sample(
            n=QUESTION_COUNT, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        embedder = Embedder()
        llm      = LLMClient()
        vs       = build_vector_store()
        split_cache: dict = {}

        print(f"\nTable 4: Uncertainty-Aware RAG Novelty Analysis")
        print(f"Questions: {QUESTION_COUNT}  |  Correct-refusal cutoff: {CORRECT_REFUSAL_CUTOFF}")
        print("=" * 80)

        for thr in to_run:
            row = run_threshold(thr, golden, vs, embedder, llm, split_cache)
            _save_checkpoint(row, existing)
    else:
        print("All thresholds already done. Regenerating observations only.")

    # Final save with refreshed observations
    ordered = [existing[t] for t in sorted(existing) if t in ALL_THRESHOLDS]
    _attach_observations(ordered)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ordered).to_csv(OUTPUT_PATH, index=False)

    df = pd.read_csv(OUTPUT_PATH)
    print(f"\n{'='*80}")
    print("Table 4: Uncertainty-Aware RAG Novelty Result")
    print(f"{'='*80}")
    pd.set_option("display.max_colwidth", 50)
    print(df.drop(columns=["key_observation"]).to_string(index=False))

    print(f"\nKey Observations")
    print("-" * 80)
    for _, r in df.iterrows():
        print(f"\n[E04 threshold={r['threshold']}]")
        print(f"  {r['key_observation']}")

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
