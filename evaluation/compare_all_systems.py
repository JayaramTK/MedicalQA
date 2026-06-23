"""
Master comparison table across all experiments.

Rows:
  E01 × 6 prompt strategies  (default, zero_shot, few_shot, cot, tot, self_consistency)
  E02 × 3 retrieval types     (Basic RAG, Sparse RAG, Hybrid RAG)
  E03 — RAG with Sentence Evidence
  E04 — RAG with Uncertainty Handling
  E05 — Final Combined RAG System

Data sources (in priority order):
  1. results/all_systems_comparison.csv  — own checkpoint (full 6 metrics, never re-runs)
  2. results/metrics_tracker.csv         — seeds E01/E02/E03/E04/E05 defaults on first use

Output: results/all_systems_comparison.csv
Usage : .venv/bin/python evaluation/compare_all_systems.py
        .venv/bin/python evaluation/compare_all_systems.py --experiments E01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from preprocessing.config import GOLDEN_CSV_PATH, RESULTS_DIR
from shared.llm_client import LLMClient
from shared.embedder   import Embedder
from shared.metrics    import compute_metrics
from experiments.e02_basic_rag.implementation.pipeline        import BasicRAGPipeline
from experiments.e02_basic_rag.implementation.sparse_pipeline import SparseRAGPipeline
from experiments.e02_basic_rag.implementation.hybrid_pipeline import HybridRAGPipeline
from experiments.e03_sentence_evidence.implementation.pipeline import SentenceEvidencePipeline
from experiments.e04_uncertainty_rag.implementation.pipeline   import UncertaintyRAGPipeline
from experiments.e05_combined_rag.implementation.pipeline      import CombinedRAGPipeline

OUTPUT_PATH  = RESULTS_DIR / "all_systems_comparison.csv"
TRACKER_PATH = RESULTS_DIR / "metrics_tracker.csv"
MODEL        = "llama-3.3-70b-versatile"
QUESTION_COUNT  = 10
RANDOM_STATE    = 42

METRIC_COLS = [
    "faithfulness", "context_relevance", "context_precision",
    "answer_relevance", "hallucination_rate", "trust_score",
]

# ---------------------------------------------------------------------------
# Master variant list  (display order = table row order)
# ---------------------------------------------------------------------------

ALL_VARIANTS: list[dict] = [
    # E01 — Baseline LLM  ×6 prompt strategies
    dict(experiment_id="E01", system_type="Baseline LLM without RAG", prompt_type="",
         llm=MODEL, embedding_model="None", vector_db="None", chunk_size=0, chunk_overlap=0),
    dict(experiment_id="E01", system_type="Baseline LLM without RAG", prompt_type="zero_shot",
         llm=MODEL, embedding_model="None", vector_db="None", chunk_size=0, chunk_overlap=0),
    dict(experiment_id="E01", system_type="Baseline LLM without RAG", prompt_type="few_shot",
         llm=MODEL, embedding_model="None", vector_db="None", chunk_size=0, chunk_overlap=0),
    dict(experiment_id="E01", system_type="Baseline LLM without RAG", prompt_type="cot",
         llm=MODEL, embedding_model="None", vector_db="None", chunk_size=0, chunk_overlap=0),
    dict(experiment_id="E01", system_type="Baseline LLM without RAG", prompt_type="tot",
         llm=MODEL, embedding_model="None", vector_db="None", chunk_size=0, chunk_overlap=0),
    dict(experiment_id="E01", system_type="Baseline LLM without RAG", prompt_type="self_consistency",
         llm=MODEL, embedding_model="None", vector_db="None", chunk_size=0, chunk_overlap=0),
    # E02 — RAG retrieval strategies
    dict(experiment_id="E02", system_type="Basic RAG", prompt_type="",
         llm=MODEL, embedding_model="all-MiniLM-L6-v2", vector_db="ChromaDB",
         chunk_size=500, chunk_overlap=100),
    dict(experiment_id="E02", system_type="Sparse RAG", prompt_type="",
         llm=MODEL, embedding_model="all-MiniLM-L6-v2", vector_db="ChromaDB",
         chunk_size=500, chunk_overlap=100),
    dict(experiment_id="E02", system_type="Hybrid RAG", prompt_type="",
         llm=MODEL, embedding_model="all-MiniLM-L6-v2", vector_db="ChromaDB",
         chunk_size=500, chunk_overlap=100),
    # E03 – E05
    dict(experiment_id="E03", system_type="RAG with Sentence Evidence", prompt_type="",
         llm=MODEL, embedding_model="all-MiniLM-L6-v2", vector_db="ChromaDB",
         chunk_size=500, chunk_overlap=100),
    dict(experiment_id="E04", system_type="RAG with Uncertainty Handling", prompt_type="",
         llm=MODEL, embedding_model="all-MiniLM-L6-v2", vector_db="ChromaDB",
         chunk_size=500, chunk_overlap=100),
    dict(experiment_id="E05", system_type="Final Combined RAG System", prompt_type="",
         llm=MODEL, embedding_model="all-MiniLM-L6-v2", vector_db="ChromaDB",
         chunk_size=500, chunk_overlap=100),
]

# Map prompt_type → LLMClient method name  (E01)
_E01_PROMPT_FN: dict[str, str] = {
    "":                 "answer_baseline",
    "zero_shot":        "answer_zero_shot",
    "few_shot":         "answer_few_shot",
    "cot":              "answer_cot",
    "tot":              "answer_tot",
    "self_consistency": "answer_self_consistency",
}

# Map system_type → Pipeline class  (E02)
_E02_PIPELINE: dict[str, type] = {
    "Basic RAG":  BasicRAGPipeline,
    "Sparse RAG": SparseRAGPipeline,
    "Hybrid RAG": HybridRAGPipeline,
}

# Map experiment_id → Pipeline class  (E03–E05)
_E_PIPELINE: dict[str, type] = {
    "E03": SentenceEvidencePipeline,
    "E04": UncertaintyRAGPipeline,
    "E05": CombinedRAGPipeline,
}

# Variants whose metrics can be seeded directly from metrics_tracker.csv
# (experiment_id, system_type) that exist as single rows in the tracker
_TRACKER_SEED_KEYS: set[tuple] = {
    ("E01", "Baseline LLM without RAG"),
    ("E02", "Basic RAG"),
    ("E03", "RAG with Sentence Evidence"),
    ("E04", "RAG with Uncertainty Handling"),
    ("E05", "Final Combined RAG System"),
}


# ---------------------------------------------------------------------------
# Composite key helpers
# ---------------------------------------------------------------------------

def _key(v: dict) -> tuple:
    return (v.get("experiment_id", ""), v.get("system_type", ""), str(v.get("prompt_type", "")))


# ---------------------------------------------------------------------------
# Load existing checkpoint rows
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict[tuple, dict]:
    if not OUTPUT_PATH.exists():
        return {}
    df = pd.read_csv(OUTPUT_PATH)
    if "prompt_type" not in df.columns:
        df["prompt_type"] = ""
    complete: dict[tuple, dict] = {}
    for _, row in df.iterrows():
        r = row.to_dict()
        if pd.notna(r.get("faithfulness")):
            complete[_key(r)] = r
    return complete


def _seed_from_tracker() -> dict[tuple, dict]:
    """Pull single-row experiments from metrics_tracker.csv."""
    if not TRACKER_PATH.exists():
        return {}
    tracker = pd.read_csv(TRACKER_PATH)
    if "prompt_type" not in tracker.columns:
        tracker["prompt_type"] = ""
    seeded: dict[tuple, dict] = {}
    for v in ALL_VARIANTS:
        eid, stype = v["experiment_id"], v["system_type"]
        ptype = str(v.get("prompt_type", ""))
        if (eid, stype) not in _TRACKER_SEED_KEYS or ptype != "":
            continue
        rows = tracker[
            (tracker["experiment_id"] == eid) &
            (tracker["system_type"]   == stype)
        ]
        if rows.empty:
            continue
        src = rows.iloc[0].to_dict()
        if not pd.notna(src.get("faithfulness")):
            continue
        merged = {**v}
        for col in METRIC_COLS:
            merged[col] = src.get(col, 0.0)
        seeded[_key(v)] = merged
    return seeded


# ---------------------------------------------------------------------------
# Run one variant — returns averaged metrics
# ---------------------------------------------------------------------------

def _run_variant(
    cfg: dict,
    golden: pd.DataFrame,
    llm: LLMClient,
    embedder: Embedder,
) -> dict[str, float]:
    eid         = cfg["experiment_id"]
    system_type = cfg["system_type"]
    prompt_type = str(cfg.get("prompt_type", ""))

    label = f"{eid} — {system_type}"
    if prompt_type:
        label += f"  [{prompt_type}]"
    print(f"\n{'='*65}\n  {label}\n{'='*65}")

    per_q: list[dict] = []

    if eid == "E01":
        fn = getattr(llm, _E01_PROMPT_FN[prompt_type])
        for i, row in golden.reset_index(drop=True).iterrows():
            q, gt = row["question"], row["ground_truth"]
            print(f"  [{i+1:02d}/{len(golden)}] {q[:65]}…")
            answer  = fn(q)
            metrics = compute_metrics(q, answer, gt, [], [], embedder, llm, False)
            per_q.append(metrics)
            print(f"       faith={metrics['faithfulness']:.3f}  "
                  f"hall={metrics['hallucination_rate']:.3f}  "
                  f"trust={metrics['trust_score']:.3f}")

    elif eid == "E02":
        pipeline = _E02_PIPELINE[system_type]()
        for i, row in golden.reset_index(drop=True).iterrows():
            q, gt = row["question"], row["ground_truth"]
            print(f"  [{i+1:02d}/{len(golden)}] {q[:65]}…")
            result  = pipeline.run(q, gt)
            metrics = {k: result[k] for k in METRIC_COLS}
            per_q.append(metrics)
            print(f"       faith={metrics['faithfulness']:.3f}  "
                  f"ctx_rel={metrics['context_relevance']:.3f}  "
                  f"hall={metrics['hallucination_rate']:.3f}  "
                  f"trust={metrics['trust_score']:.3f}")

    else:
        pipeline = _E_PIPELINE[eid]()
        for i, row in golden.reset_index(drop=True).iterrows():
            q, gt = row["question"], row["ground_truth"]
            print(f"  [{i+1:02d}/{len(golden)}] {q[:65]}…")
            result  = pipeline.run(q, gt)
            metrics = {k: result[k] for k in METRIC_COLS}
            per_q.append(metrics)
            print(f"       faith={metrics['faithfulness']:.3f}  "
                  f"ctx_rel={metrics['context_relevance']:.3f}  "
                  f"hall={metrics['hallucination_rate']:.3f}  "
                  f"trust={metrics['trust_score']:.3f}")

    avg = {col: round(sum(m[col] for m in per_q) / len(per_q), 4) for col in METRIC_COLS}
    print(f"\n  ── Averages ──")
    for k, v_avg in avg.items():
        print(f"     {k:<22} {v_avg:.4f}")
    return avg


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def _save(completed: dict[tuple, dict]) -> None:
    ordered = [completed[_key(v)] for v in ALL_VARIANTS if _key(v) in completed]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ordered).to_csv(OUTPUT_PATH, index=False)
    print(f"  → Checkpoint saved → {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+",
                        choices=["E01", "E02", "E03", "E04", "E05"],
                        help="Run only specific experiment IDs (default: all)")
    args = parser.parse_args()

    selected_ids = set(args.experiments or ["E01", "E02", "E03", "E04", "E05"])
    variants     = [v for v in ALL_VARIANTS if v["experiment_id"] in selected_ids]

    # ── Load existing data ──────────────────────────────────────────────────
    completed = _load_checkpoint()

    # Seed from metrics_tracker for experiments not yet in checkpoint
    for k, row in _seed_from_tracker().items():
        if k not in completed:
            completed[k] = row
            print(f"  Seeded from tracker: {k[0]} {k[1]}")

    to_run  = [v for v in variants if _key(v) not in completed]
    skipped = [v for v in variants if _key(v) in completed]

    if skipped:
        print(f"\nSkipping {len(skipped)} already-completed variant(s):")
        for v in skipped:
            tag = f"  [{v['prompt_type']}]" if v["prompt_type"] else ""
            print(f"  {v['experiment_id']}  {v['system_type']}{tag}")

    if not to_run:
        print("\nAll selected variants already completed.")
    else:
        print(f"\nRunning {len(to_run)} variant(s) on {QUESTION_COUNT} questions.")
        golden   = pd.read_csv(GOLDEN_CSV_PATH).sample(
            n=QUESTION_COUNT, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        llm      = LLMClient()
        embedder = Embedder()

        for cfg in to_run:
            avg = _run_variant(cfg, golden, llm, embedder)
            completed[_key(cfg)] = {**cfg, **avg}
            _save(completed)

    # ── Final display ───────────────────────────────────────────────────────
    _save(completed)   # refresh observations order in file
    df = pd.read_csv(OUTPUT_PATH)

    display_cols = [
        "experiment_id", "system_type", "prompt_type",
        "embedding_model", "vector_db", "chunk_size", "chunk_overlap",
        "faithfulness", "context_relevance", "context_precision",
        "answer_relevance", "hallucination_rate", "trust_score",
    ]
    # keep only columns that exist in the file
    display_cols = [c for c in display_cols if c in df.columns]

    print(f"\n{'='*100}")
    print("All Systems Comparison Table")
    print(f"{'='*100}")
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.width", 200)
    print(df[display_cols].to_string(index=False))
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
