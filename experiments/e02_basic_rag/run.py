"""Interactive single-query CLI for E02 — Basic RAG."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from implementation.pipeline import BasicRAGPipeline

if __name__ == "__main__":
    question = input("Enter your medical question: ").strip()
    pipeline = BasicRAGPipeline()
    result   = pipeline.run(question)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")
    print(f"Metrics: faithfulness={result['faithfulness']}  "
          f"hallucination={result['hallucination_rate']}  trust={result['trust_score']}")
