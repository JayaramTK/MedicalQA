"""
Generate Table 4: Uncertainty-Aware RAG Novelty Result.

Rows produced (one per threshold):
  E04 — threshold 0.82   (few refusals)
  E04 — threshold 0.86   (moderate refusals)
  E04 — threshold 0.90   (many refusals)

NOTE ON THRESHOLDS
  The original design thresholds (0.60, 0.65, 0.70) assume retrieval scores
  in the 0.3–0.7 range. In this project, all-MiniLM-L6-v2 + ChromaDB returns
  similarity scores between 0.78 and 0.91 for medical questions against this
  knowledge base, so none of the design thresholds would trigger a refusal.
  The calibrated thresholds above cover the actual score distribution and
  produce meaningful variation. Change THRESHOLDS below if needed.

Metrics:
  Answered Questions  — questions where max_score >= threshold (answered normally)
  Refused  Questions  — questions where max_score <  threshold (refused)
  Correct Refusal     — refused AND max_score < CORRECT_REFUSAL_CUTOFF
                        (score was genuinely too low to trust the context)
  Wrong Refusal       — refused AND max_score >= CORRECT_REFUSAL_CUTOFF
                        (score was borderline; system was over-cautious)
  Refusal Accuracy    — Correct Refusals / max(Refused, 1)
  Hallucination Rate  — computed only over answered questions
  Trust Score         — 0.4*(1-hallucination_rate) + 0.3*refusal_accuracy
                        + 0.3*(answered/total)

Output: data/experiments/uncertainty_rag_analysis.csv
Usage : .venv/bin/python scripts/uncertainty_rag_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medqa.config import DEFAULT_CONFIG
from medqa.experiments import BaseExperiment, TOP_K, QuestionResult
from medqa.rag.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Configuration — adjust thresholds here if needed
# ---------------------------------------------------------------------------

THRESHOLDS: list[float] = [0.82, 0.86, 0.90]

# Below this → refusal was CORRECT (context genuinely too weak to trust)
# Between this and the threshold → refusal was WRONG (over-cautious)
CORRECT_REFUSAL_CUTOFF: float = 0.84

QUESTION_COUNT = 10
OUTPUT_PATH    = ROOT / "data" / "experiments" / "uncertainty_rag_analysis.csv"


# ---------------------------------------------------------------------------
# E04 variant with configurable threshold + actual refusal logic
# ---------------------------------------------------------------------------

class E04_ThresholdVariant(BaseExperiment):
    """E04 re-parameterised with a configurable high-confidence threshold.

    Questions whose max retrieval score falls below the threshold are
    *refused* — no answer is generated and they count toward Refused Questions.
    """

    def __init__(self, knowledge_base: list[dict], threshold: float) -> None:
        super().__init__(knowledge_base)
        self.threshold = threshold

    def setup(self) -> None:
        self.vector_store = self._build_vector_store()

    def run_question(
        self, question: str, ground_truth: str
    ) -> tuple[QuestionResult | None, float, bool]:
        """Return (result_or_None, max_retrieval_score, was_refused)."""
        hits   = self.vector_store.query(question, n_results=TOP_K)
        chunks = [h["text"] for h in hits]
        scores = [h["score"] for h in hits]
        max_score = max(scores) if scores else 0.0

        refused = max_score < self.threshold
        if refused:
            return None, max_score, True

        # Answered — generate response and compute metrics
        context = "\n\n".join(chunks)
        answer  = self.llm.answer_uncertainty(
            question, context,
            confidence="High" if max_score >= self.threshold else "Medium",
        )
        metrics = compute_metrics(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            retrieved_chunks=chunks,
            retrieval_scores=scores,
            embedder=self.embedder,
            llm=self.llm,
            has_rag=True,
        )
        result = QuestionResult(
            question=question, answer=answer, metrics=metrics,
            chunks=chunks, scores=scores,
        )
        return result, max_score, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_golden() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_CONFIG["processed_csv_path"])
    return df.sample(n=QUESTION_COUNT, random_state=42).reset_index(drop=True)


def load_knowledge_base() -> list[dict]:
    kb_path = DEFAULT_CONFIG["knowledge_db_path"].parent / "knowledge_base.csv"
    return pd.read_csv(kb_path).to_dict(orient="records")


def run_threshold_variant(
    threshold: float,
    questions: pd.DataFrame,
    kb_docs: list[dict],
) -> dict:
    """Run E04 with one threshold and return all table metrics."""
    print(f"\n  Running E04 (threshold={threshold}) on {len(questions)} questions…")
    exp = E04_ThresholdVariant(kb_docs, threshold=threshold)
    exp.setup()

    total      = len(questions)
    answered_q : list[dict] = []
    refused_scores: list[float] = []

    for i, row in questions.iterrows():
        q, gt = row["question"], row["ground_truth"]
        result, max_score, refused = exp.run_question(q, gt)

        if refused:
            refused_scores.append(max_score)
            print(f"    [{i+1:02d}/{total}] REFUSED  max_score={max_score:.4f} < {threshold}")
        else:
            answered_q.append(result.metrics)
            print(f"    [{i+1:02d}/{total}] Answered max_score={max_score:.4f}  "
                  f"faith={result.metrics['faithfulness']:.3f}  "
                  f"hall={result.metrics['hallucination_rate']:.3f}")

    answered   = len(answered_q)
    refused    = len(refused_scores)

    correct_refusals = sum(1 for s in refused_scores if s < CORRECT_REFUSAL_CUTOFF)
    wrong_refusals   = refused - correct_refusals
    refusal_accuracy = round(correct_refusals / max(refused, 1), 4)

    if answered_q:
        hallucination_rate = round(
            sum(m["hallucination_rate"] for m in answered_q) / answered, 4
        )
    else:
        hallucination_rate = 1.0   # refused all → worst case

    trust_score = round(
        0.4 * (1 - hallucination_rate)
        + 0.3 * refusal_accuracy
        + 0.3 * (answered / total),
        4,
    )

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


def key_observation(row: dict, rows: list[dict]) -> str:
    thr   = row["threshold"]
    ref   = row["refused_questions"]
    total = row["total_questions"]
    acc   = row["refusal_accuracy"]
    hall  = row["hallucination_rate"]

    if ref == 0:
        return (
            f"At threshold={thr}, all {total} questions are answered "
            f"(retrieval scores exceed threshold). Hallucination rate: {hall:.2f}. "
            f"No refusals triggered."
        )

    coverage = round(row["answered_questions"] / total, 2)
    cr_pct   = f"{row['correct_refusals']}/{ref}" if ref else "0/0"
    return (
        f"Threshold={thr} refuses {ref}/{total} questions "
        f"({ref/total:.0%} refusal rate). "
        f"Coverage: {coverage:.0%}. "
        f"Correct refusals: {cr_pct} (accuracy={acc:.2f}). "
        f"Hallucination rate on answered questions: {hall:.2f}. "
        f"{'Stricter threshold reduces hallucination but limits coverage.' if thr == max(r['threshold'] for r in rows) else 'Balances coverage and confidence.'}"
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers — save after every threshold so progress isn't lost
# ---------------------------------------------------------------------------

def _load_existing() -> dict[float, dict]:
    """Return COMPLETE rows keyed by threshold.

    A row is considered complete only if hallucination_rate is a valid number.
    Rows with blank/NaN metric columns are treated as incomplete and will be re-run.
    """
    if not OUTPUT_PATH.exists():
        return {}
    df = pd.read_csv(OUTPUT_PATH)
    complete: dict[float, dict] = {}
    for _, row in df.iterrows():
        val = row.get("hallucination_rate", None)
        try:
            if val is not None and str(val).strip() not in ("", "nan"):
                float(val)          # confirms it's a real number
                complete[float(row["threshold"])] = row.to_dict()
        except (ValueError, TypeError):
            pass   # incomplete row — will be re-run
    return complete


def _save_row(row: dict, all_rows: list[dict]) -> None:
    """Append / overwrite this threshold's row and rewrite the CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing()
    existing[row["threshold"]] = row
    # Write rows in threshold order
    ordered = [existing[t] for t in sorted(existing)]
    # Attach key_observation now that we have all saved rows
    for r in ordered:
        if "key_observation" not in r or not r["key_observation"]:
            r["key_observation"] = key_observation(r, ordered)
    pd.DataFrame(ordered).to_csv(OUTPUT_PATH, index=False)
    print(f"  → Checkpoint saved → {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds", nargs="+", type=float, metavar="T",
        help="Run only these threshold values, e.g. --thresholds 0.90",
    )
    args = parser.parse_args()

    selected = args.thresholds if args.thresholds else THRESHOLDS

    # Skip thresholds that are already fully computed in the CSV
    existing  = _load_existing()
    to_run    = [t for t in selected if t not in existing]
    skipped   = [t for t in selected if t in existing]

    if skipped:
        print(f"\nSkipping already-completed thresholds: {skipped}")
    if to_run:
        print(f"Will run thresholds: {to_run}")
    if not to_run:
        print("All selected thresholds already done. Showing saved results.")
    else:
        questions = load_golden()
        kb_docs   = load_knowledge_base()

        print(f"\nTable 4: Uncertainty-Aware RAG Novelty Analysis")
        print(f"Thresholds to run : {to_run}")
        print(f"Questions  : {QUESTION_COUNT}  |  Knowledge base: {len(kb_docs)} docs")
        print(f"Correct-refusal cutoff: {CORRECT_REFUSAL_CUTOFF}")
        print("=" * 70)

        for thr in to_run:
            row = run_threshold_variant(thr, questions, kb_docs)
            _save_row(row, [])   # checkpoint immediately

    # Final display from saved file
    df = pd.read_csv(OUTPUT_PATH)
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    pd.set_option("display.max_colwidth", 50)
    print(df.drop(columns=["key_observation"]).to_string(index=False))

    print(f"\nKey Observations")
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"\n[E04 threshold={r['threshold']}]")
        print(f"  {r['key_observation']}")

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
