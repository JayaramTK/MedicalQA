"""Interactive single-query CLI for E01 — Baseline LLM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from implementation.pipeline import BaselinePipeline

if __name__ == "__main__":
    question = input("Enter your medical question: ").strip()
    if not question:
        print("No question provided.")
        sys.exit(1)
    result = BaselinePipeline().run(question)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nMetrics: faithfulness={result['faithfulness']}  "
          f"hallucination={result['hallucination_rate']}  "
          f"trust={result['trust_score']}")
