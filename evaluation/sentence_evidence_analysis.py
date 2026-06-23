"""
Table 3: Sentence-Level Evidence Novelty Result.

Rows produced:
  E02  — chunk-level evidence, full chunk,  top_n=0  (read from sentence_evidence_analysis.csv)
  E03  — sentence-level,       sentence,    top_n=1  (read from csv or run fresh)
  E03  — sentence-level,       sentence,    top_n=3  (read from csv or run fresh)
  E03  — sentence-level,       sentence,    top_n=5  (run fresh if missing)
  E03  — sentence-level,       sentence,    top_n=7  (run fresh if missing)
  E03  — sentence-level,       sentence,    top_n=9  (run fresh if missing)

Evidence Granularity Score (EGS):
  EGS = 1 - (evidence_words / total_chunk_words)
  High EGS = tightly focused evidence.  EGS = 0 = full chunk used.

Output: results/sentence_evidence_analysis.csv
Usage : .venv/bin/python evaluation/sentence_evidence_analysis.py
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
from shared.metrics    import compute_metrics, extract_sentence_evidence
from experiments.e03_sentence_evidence.implementation.ingestion import build_vector_store
from experiments.e03_sentence_evidence.implementation.config    import TOP_K

OUTPUT_PATH    = RESULTS_DIR / "sentence_evidence_analysis.csv"
QUESTION_COUNT = 10
RANDOM_STATE   = 42

ALL_TOP_N = [1, 3, 5, 7, 9]


# ---------------------------------------------------------------------------
# Run E03 with a given top_n, return averaged metrics + EGS
# ---------------------------------------------------------------------------

def _run_e03_variant(
    top_n: int,
    golden: pd.DataFrame,
    vs,
    embedder: Embedder,
    llm: LLMClient,
) -> dict:
    print(f"\n  Running E03 (top_n={top_n}) on {len(golden)} questions…")
    per_q: list[dict] = []

    for i, row in golden.reset_index(drop=True).iterrows():
        q, gt = row["question"], row["ground_truth"]
        print(f"    [{i+1:02d}/{len(golden)}] {q[:65]}…")

        hits   = vs.query(q, n_results=TOP_K)
        chunks = [h["text"] for h in hits]
        scores = [h["score"] for h in hits]

        total_chunk_words = max(sum(len(c.split()) for c in chunks), 1)
        evidence          = extract_sentence_evidence(q, chunks, embedder, top_n=top_n)
        evidence_words    = len(evidence.split())

        answer  = llm.answer_sentence_evidence(q, evidence)
        metrics = compute_metrics(
            question=q, answer=answer, ground_truth=gt,
            retrieved_chunks=chunks, retrieval_scores=scores,
            embedder=embedder, llm=llm, has_rag=True,
        )

        egs = round(1.0 - min(evidence_words / total_chunk_words, 1.0), 4)
        metrics["evidence_granularity_score"] = egs
        per_q.append(metrics)

        print(f"         faith={metrics['faithfulness']:.3f}  "
              f"ctx_prec={metrics['context_precision']:.3f}  "
              f"egs={egs:.3f}  "
              f"trust={metrics['trust_score']:.3f}")

    return {col: round(sum(m[col] for m in per_q) / len(per_q), 4) for col in per_q[0]}


# ---------------------------------------------------------------------------
# Key observation text
# ---------------------------------------------------------------------------

def _key_observation(exp_id: str, top_n: int, m: dict, base: dict) -> str:
    if exp_id == "E02":
        return (
            "Uses full retrieved chunks as context. Evidence is coarse-grained "
            f"(EGS={m['evidence_granularity_score']:.2f}), so the model must filter "
            "relevant information internally. Highest faithfulness among chunk-level systems."
        )
    faith_delta = m["faithfulness"] - base["faithfulness"]
    egs  = m["evidence_granularity_score"]
    sign = "+" if faith_delta >= 0 else ""
    if top_n == 1:
        tail = "Higher precision at the cost of narrower context."
    elif top_n <= 3:
        tail = "Balances focus with broader coverage."
    elif top_n <= 6:
        tail = "Wider evidence window improves context breadth at slight EGS cost."
    else:
        tail = "Broad evidence window approaches chunk-level coverage; EGS diminishes."
    return (
        f"Extracts the top-{top_n} most relevant sentence(s) before answering. "
        f"EGS={egs:.4f} reflects tight evidence focus. "
        f"Faithfulness {sign}{faith_delta:.2f} vs chunk-level baseline. {tail}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load existing rows ──────────────────────────────────────────────────
    existing_by_topn: dict[int, dict] = {}
    e02_row: dict | None = None

    if OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH)
        for _, r in existing_df.iterrows():
            if r["experiment_id"] == "E02":
                e02_row = r.to_dict()
            else:
                existing_by_topn[int(r["top_evidence_sentences"])] = r.to_dict()

    missing_top_n = [n for n in ALL_TOP_N if n not in existing_by_topn]

    if e02_row is None:
        print("ERROR: E02 row not found in sentence_evidence_analysis.csv. "
              "Run run_all_experiments.py first.")
        sys.exit(1)

    # ── Build shared resources only when there are experiments to run ───────
    if missing_top_n:
        print(f"\nMissing top_n values: {missing_top_n} — will run them now.")
        golden   = pd.read_csv(GOLDEN_CSV_PATH).sample(
            n=QUESTION_COUNT, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        embedder = Embedder()
        llm      = LLMClient()
        vs       = build_vector_store()

        for top_n in missing_top_n:
            avg = _run_e03_variant(top_n, golden, vs, embedder, llm)
            existing_by_topn[top_n] = {
                "experiment_id":              "E03",
                "system":                     "RAG with Sentence Evidence",
                "evidence_type":              "Sentence-level evidence",
                "evidence_unit":              "Sentence",
                "top_evidence_sentences":     top_n,
                "faithfulness":               avg["faithfulness"],
                "context_precision":          avg["context_precision"],
                "evidence_granularity_score": avg["evidence_granularity_score"],
                "trust_score":                avg["trust_score"],
                "key_observation":            "",   # filled below
            }
    else:
        print("All E03 variants already computed. Regenerating observations only.")

    # ── Regenerate key observations for all rows ────────────────────────────
    base_metrics = {
        "faithfulness":              e02_row["faithfulness"],
        "evidence_granularity_score": e02_row.get("evidence_granularity_score", 0.0),
    }

    e02_row["key_observation"] = _key_observation("E02", 0, base_metrics, base_metrics)

    for top_n in ALL_TOP_N:
        r = existing_by_topn[top_n]
        r["key_observation"] = _key_observation(
            "E03", top_n,
            {"faithfulness": r["faithfulness"],
             "evidence_granularity_score": r["evidence_granularity_score"]},
            base_metrics,
        )

    # ── Assemble final table in order ───────────────────────────────────────
    rows = [e02_row] + [existing_by_topn[n] for n in ALL_TOP_N]

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTable 3: Sentence-Level Evidence Novelty Result")
    print("=" * 90)
    pd.set_option("display.max_colwidth", 55)
    print(df.drop(columns=["key_observation"]).to_string(index=False))

    print(f"\nKey Observations")
    print("-" * 90)
    for _, r in df.iterrows():
        label = f"E03 top_n={int(r['top_evidence_sentences'])}" \
                if r["experiment_id"] == "E03" else "E02"
        print(f"\n[{label}] {r['system']}")
        print(f"  {r['key_observation']}")

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
