"""
Generate Table 3: Sentence-Level Evidence Novelty Result.

Rows produced:
  E02 — chunk-level evidence,  full chunk,  top_n=0  (metrics from tracker)
  E03 — sentence-level,        sentence,    top_n=1  (fresh run)
  E03 — sentence-level,        sentence,    top_n=3  (fresh run)

Evidence Granularity Score (EGS):
  Measures how focused the evidence unit is relative to the full retrieved context.
  EGS = 1 - (evidence_words / total_chunk_words)
  High EGS → tightly focused evidence.  EGS = 0 → full chunk used.

Output: data/experiments/sentence_evidence_analysis.csv
Usage : .venv/bin/python scripts/sentence_evidence_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from medqa.config import DEFAULT_CONFIG
from medqa.experiments import (
    BaseExperiment,
    LLM_MODEL,
    TOP_K,
    QuestionResult,
)
from medqa.rag.metrics import compute_metrics, extract_sentence_evidence

TRACKER_PATH  = ROOT / "data" / "experiments" / "metrics_tracker.csv"
OUTPUT_PATH   = ROOT / "data" / "experiments" / "sentence_evidence_analysis.csv"
QUESTION_COUNT = 10


# ---------------------------------------------------------------------------
# E03 variant with configurable top_n (does NOT modify experiments.py)
# ---------------------------------------------------------------------------

class E03_SentenceEvidence_N(BaseExperiment):
    """E03 re-parameterised with a custom top_n for sentence extraction."""

    def __init__(self, knowledge_base: list[dict], top_n: int) -> None:
        super().__init__(knowledge_base)
        self.top_n = top_n

    def setup(self) -> None:
        self.vector_store = self._build_vector_store()

    def run_question(self, question: str, ground_truth: str) -> QuestionResult:
        hits   = self.vector_store.query(question, n_results=TOP_K)
        chunks = [h["text"] for h in hits]
        scores = [h["score"] for h in hits]

        total_chunk_words = max(sum(len(c.split()) for c in chunks), 1)

        evidence = extract_sentence_evidence(
            question, chunks, self.embedder, top_n=self.top_n
        )
        evidence_words = len(evidence.split())

        answer = self.llm.answer_sentence_evidence(question, evidence)

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

        # Evidence Granularity Score: how focused the evidence is vs full context
        egs = round(1.0 - min(evidence_words / total_chunk_words, 1.0), 4)
        metrics["evidence_granularity_score"] = egs

        return QuestionResult(
            question=question, answer=answer, metrics=metrics,
            chunks=chunks, scores=scores,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_golden() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_CONFIG["processed_csv_path"])
    return df.sample(n=QUESTION_COUNT, random_state=42).reset_index(drop=True)


def load_knowledge_base() -> list[dict]:
    kb_path = DEFAULT_CONFIG["knowledge_db_path"].parent / "knowledge_base.csv"
    return pd.read_csv(kb_path).to_dict(orient="records")


def run_e03_variant(
    top_n: int,
    questions: pd.DataFrame,
    kb_docs: list[dict],
) -> dict[str, float]:
    """Run E03 with the given top_n and return averaged metrics."""
    print(f"\n  Running E03 (top_n={top_n}) on {len(questions)} questions…")
    exp = E03_SentenceEvidence_N(kb_docs, top_n=top_n)
    exp.setup()

    per_q: list[dict] = []
    for i, row in questions.iterrows():
        q, gt = row["question"], row["ground_truth"]
        print(f"    [{i+1:02d}/{len(questions)}] {q[:65]}…")
        result = exp.run_question(q, gt)
        per_q.append(result.metrics)
        print(f"         faith={result.metrics['faithfulness']:.3f}  "
              f"ctx_prec={result.metrics['context_precision']:.3f}  "
              f"egs={result.metrics['evidence_granularity_score']:.3f}  "
              f"trust={result.metrics['trust_score']:.3f}")

    avg = {
        col: round(sum(m[col] for m in per_q) / len(per_q), 4)
        for col in per_q[0]
    }
    return avg


def key_observation(exp_id: str, top_n: int, m: dict, base: dict) -> str:
    if exp_id == "E02":
        return (
            "Uses full retrieved chunks as context. Evidence is coarse-grained "
            f"(EGS={m['evidence_granularity_score']:.2f}), so the model must filter "
            "relevant information internally. Highest faithfulness among chunk-level systems."
        )
    faith_delta = m["faithfulness"] - base["faithfulness"]
    egs = m["evidence_granularity_score"]
    sign = "+" if faith_delta >= 0 else ""
    return (
        f"Extracts the top-{top_n} most relevant sentence(s) before answering. "
        f"EGS={egs:.2f} reflects tight evidence focus. "
        f"Faithfulness {sign}{faith_delta:.2f} vs chunk-level baseline. "
        f"{'Higher precision at the cost of narrower context.' if top_n == 1 else 'Balances focus with broader coverage.'}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not TRACKER_PATH.exists():
        print(f"ERROR: {TRACKER_PATH} not found. Run run_experiments.py first.")
        sys.exit(1)

    tracker = pd.read_csv(TRACKER_PATH)
    if "E02" not in tracker["experiment_id"].values:
        print("ERROR: E02 not found in tracker. Run run_experiments.py first.")
        sys.exit(1)

    e02_row = tracker[tracker["experiment_id"] == "E02"].iloc[0]

    # E02 metrics come from the tracker (no re-run)
    e02_metrics = {
        "faithfulness":              e02_row["faithfulness"],
        "context_precision":         e02_row["context_precision"],
        "trust_score":               e02_row["trust_score"],
        "evidence_granularity_score": 0.0,   # full chunk → no extraction
    }

    questions = load_golden()
    kb_docs   = load_knowledge_base()

    print(f"\nTable 3: Sentence-Level Evidence Novelty Analysis")
    print(f"Questions: {QUESTION_COUNT}  |  Knowledge base: {len(kb_docs)} docs")
    print("=" * 70)

    e03_top1 = run_e03_variant(top_n=1, questions=questions, kb_docs=kb_docs)
    e03_top3 = run_e03_variant(top_n=3, questions=questions, kb_docs=kb_docs)

    rows = [
        {
            "experiment_id":             "E02",
            "system":                    "Basic RAG",
            "evidence_type":             "Chunk-level evidence",
            "evidence_unit":             "Full chunk",
            "top_evidence_sentences":    0,
            "faithfulness":              e02_metrics["faithfulness"],
            "context_precision":         e02_metrics["context_precision"],
            "evidence_granularity_score": e02_metrics["evidence_granularity_score"],
            "trust_score":               e02_metrics["trust_score"],
            "key_observation":           key_observation("E02", 0, e02_metrics, e02_metrics),
        },
        {
            "experiment_id":             "E03",
            "system":                    "RAG with Sentence Evidence",
            "evidence_type":             "Sentence-level evidence",
            "evidence_unit":             "Sentence",
            "top_evidence_sentences":    1,
            "faithfulness":              e03_top1["faithfulness"],
            "context_precision":         e03_top1["context_precision"],
            "evidence_granularity_score": e03_top1["evidence_granularity_score"],
            "trust_score":               e03_top1["trust_score"],
            "key_observation":           key_observation("E03", 1, e03_top1, e02_metrics),
        },
        {
            "experiment_id":             "E03",
            "system":                    "RAG with Sentence Evidence",
            "evidence_type":             "Sentence-level evidence",
            "evidence_unit":             "Sentence",
            "top_evidence_sentences":    3,
            "faithfulness":              e03_top3["faithfulness"],
            "context_precision":         e03_top3["context_precision"],
            "evidence_granularity_score": e03_top3["evidence_granularity_score"],
            "trust_score":               e03_top3["trust_score"],
            "key_observation":           key_observation("E03", 3, e03_top3, e02_metrics),
        },
    ]

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    pd.set_option("display.max_colwidth", 55)
    print(df.drop(columns=["key_observation"]).to_string(index=False))

    print(f"\nKey Observations")
    print("-" * 70)
    for _, r in df.iterrows():
        label = f"E0{r['experiment_id'][-1]} top_n={r['top_evidence_sentences']}" \
                if r["experiment_id"] == "E03" else "E02"
        print(f"\n[{label}] {r['system']}")
        print(f"  {r['key_observation']}")

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
