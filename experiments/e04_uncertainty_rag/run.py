"""Interactive single-query CLI for e04_uncertainty_rag."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from implementation.pipeline import UncertaintyRAGPipeline
if __name__ == "__main__":
    q = input("Enter your medical question: ").strip()
    r = UncertaintyRAGPipeline().run(q)
    print(f"\nAnswer:\n{r['answer']}")
    print(f"Metrics: faithfulness={r['faithfulness']}  hallucination={r['hallucination_rate']}  trust={r['trust_score']}")
