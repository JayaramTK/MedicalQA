"""
Evaluation orchestrator — runs all five experiments and updates results/metrics_tracker.csv.

Usage:
    python evaluation/run_all_experiments.py                     # 10 questions, all 5
    python evaluation/run_all_experiments.py --sample 20         # custom sample
    python evaluation/run_all_experiments.py --experiments E01 E02  # subset
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from preprocessing.config import GOLDEN_CSV_PATH, RESULTS_DIR, KNOWLEDGE_BASE_CSV
from shared.llm_client import LLMClient
from shared.embedder   import Embedder
from shared.chunker    import chunk_documents
from shared.vectorstore import VectorStore
from shared.metrics    import compute_metrics
from experiments.e02_basic_rag.implementation.pipeline         import BasicRAGPipeline
from experiments.e03_sentence_evidence.implementation.pipeline import SentenceEvidencePipeline
from experiments.e04_uncertainty_rag.implementation.pipeline   import UncertaintyRAGPipeline
from experiments.e05_combined_rag.implementation.pipeline      import CombinedRAGPipeline

METRIC_COLS = ["faithfulness","context_relevance","context_precision",
               "answer_relevance","hallucination_rate","trust_score"]


def _update_tracker(exp_config: dict, avg: dict[str, float]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "metrics_tracker.csv"
    row  = {**exp_config, **avg}
    if path.exists():
        df   = pd.read_csv(path)
        mask = df["experiment_id"] == exp_config["experiment_id"]
        if mask.any():
            for k, v in row.items():
                df.loc[mask, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
    print(f"  → Tracker updated: {path}")


def _build_store(kb_path) -> VectorStore:
    kb     = pd.read_csv(kb_path).to_dict(orient="records")
    chunks = chunk_documents(kb, 500, 100)
    vs     = VectorStore()
    vs.add_chunks(chunks)
    return vs


def run_experiment(eid: str, questions: pd.DataFrame,
                   llm: LLMClient, embedder: Embedder) -> dict[str, float]:
    PIPELINES = {
        "E02": BasicRAGPipeline,
        "E03": SentenceEvidencePipeline,
        "E04": UncertaintyRAGPipeline,
        "E05": CombinedRAGPipeline,
    }
    CONFIGS = {
        "E01": dict(experiment_id="E01", system_type="Baseline LLM without RAG",
                    llm="llama-3.3-70b-versatile", embedding_model="None",
                    vector_db="None", chunk_size=0, chunk_overlap=0),
        "E02": dict(experiment_id="E02", system_type="Basic RAG",
                    llm="llama-3.3-70b-versatile", embedding_model="all-MiniLM-L6-v2",
                    vector_db="ChromaDB", chunk_size=500, chunk_overlap=100),
        "E03": dict(experiment_id="E03", system_type="RAG with Sentence Evidence",
                    llm="llama-3.3-70b-versatile", embedding_model="all-MiniLM-L6-v2",
                    vector_db="ChromaDB", chunk_size=500, chunk_overlap=100),
        "E04": dict(experiment_id="E04", system_type="RAG with Uncertainty Handling",
                    llm="llama-3.3-70b-versatile", embedding_model="all-MiniLM-L6-v2",
                    vector_db="ChromaDB", chunk_size=500, chunk_overlap=100),
        "E05": dict(experiment_id="E05", system_type="Final Combined RAG System",
                    llm="llama-3.3-70b-versatile", embedding_model="all-MiniLM-L6-v2",
                    vector_db="ChromaDB", chunk_size=500, chunk_overlap=100),
    }

    pipeline = PIPELINES[eid]() if eid in PIPELINES else None
    cfg      = CONFIGS[eid]
    per_q: list[dict] = []

    print(f"\n{'='*60}\n  {eid} — {cfg['system_type']}\n{'='*60}")

    for i, row in questions.reset_index(drop=True).iterrows():
        q, gt = row["question"], row["ground_truth"]
        print(f"  [{i+1:02d}/{len(questions)}] {q[:65]}…")
        if eid == "E01":
            answer  = llm.answer_baseline(q)
            metrics = compute_metrics(q, answer, gt, [], [], embedder, llm, False)
        else:
            result  = pipeline.run(q, gt)
            metrics = {k: result[k] for k in METRIC_COLS}
        per_q.append(metrics)
        for k, v in metrics.items():
            print(f"       {k:<22} {v:.4f}")

    avg = {col: round(sum(m[col] for m in per_q) / len(per_q), 4) for col in METRIC_COLS}
    print(f"\n  ── Averages ──")
    for k, v in avg.items():
        print(f"     {k:<22} {v:.4f}")
    _update_tracker(cfg, avg)
    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--sample", type=int, default=10)
    group.add_argument("--all",    action="store_true")
    parser.add_argument("--experiments", nargs="+",
                        choices=["E01","E02","E03","E04","E05"])
    args = parser.parse_args()

    golden   = pd.read_csv(GOLDEN_CSV_PATH)
    if not args.all:
        golden = golden.sample(n=args.sample, random_state=42).reset_index(drop=True)

    selected = args.experiments or ["E01","E02","E03","E04","E05"]
    print(f"\nRunning {len(selected)} experiment(s) on {len(golden)} questions.")

    llm      = LLMClient()
    embedder = Embedder()

    for eid in selected:
        run_experiment(eid, golden, llm, embedder)


    print("\n\n── Final Metrics Tracker ──")
    print(pd.read_csv(RESULTS_DIR / "metrics_tracker.csv").to_string(index=False))


if __name__ == "__main__":
    main()
