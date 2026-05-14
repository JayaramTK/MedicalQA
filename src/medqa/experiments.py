"""
Five RAG experiment implementations (E01-E05).

Each experiment class exposes:
    setup()             — one-time initialisation (build vector store, etc.)
    run_question()      — generate answer + compute metrics for one question
    CONFIG              — dict with system metadata for the tracker
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rag.chunker import chunk_documents
from .rag.embedder import Embedder
from .rag.llm import LLMClient
from .rag.metrics import compute_metrics, extract_sentence_evidence
from .rag.vectorstore import VectorStore

LLM_MODEL     = "llama-3.3-70b-versatile"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 3          # chunks retrieved per query


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    question:   str
    answer:     str
    metrics:    dict[str, float]
    chunks:     list[str] = field(default_factory=list)
    scores:     list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseExperiment:
    CONFIG: dict = {}

    def __init__(self, knowledge_base: list[dict]) -> None:
        self.knowledge_base = knowledge_base
        self.llm = LLMClient(model=LLM_MODEL)
        self.embedder = Embedder()
        self.vector_store: VectorStore | None = None

    def _build_vector_store(self) -> VectorStore:
        vs = VectorStore()
        chunks = chunk_documents(
            self.knowledge_base, CHUNK_SIZE, CHUNK_OVERLAP
        )
        vs.add_chunks(chunks)
        return vs

    def setup(self) -> None:
        """Override to perform one-time setup (e.g. build vector store)."""

    def run_question(
        self, question: str, ground_truth: str
    ) -> QuestionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# E01 – Baseline LLM (no RAG)
# ---------------------------------------------------------------------------

class E01_BaselineLLM(BaseExperiment):
    CONFIG = {
        "experiment_id": "E01",
        "system_type":   "Baseline LLM without RAG",
        "llm":           LLM_MODEL,
        "embedding_model": "None",
        "vector_db":     "None",
        "chunk_size":    0,
        "chunk_overlap": 0,
    }

    def run_question(self, question: str, ground_truth: str) -> QuestionResult:
        answer = self.llm.answer_baseline(question)
        metrics = compute_metrics(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            retrieved_chunks=[],
            retrieval_scores=[],
            embedder=self.embedder,
            llm=self.llm,
            has_rag=False,
        )
        return QuestionResult(question=question, answer=answer, metrics=metrics)


# ---------------------------------------------------------------------------
# E02 – Basic RAG
# ---------------------------------------------------------------------------

class E02_BasicRAG(BaseExperiment):
    CONFIG = {
        "experiment_id": "E02",
        "system_type":   "Basic RAG",
        "llm":           LLM_MODEL,
        "embedding_model": EMBED_MODEL,
        "vector_db":     "ChromaDB",
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    def setup(self) -> None:
        self.vector_store = self._build_vector_store()

    def run_question(self, question: str, ground_truth: str) -> QuestionResult:
        hits = self.vector_store.query(question, n_results=TOP_K)
        chunks = [h["text"] for h in hits]
        scores = [h["score"] for h in hits]
        context = "\n\n".join(chunks)

        answer = self.llm.answer_rag(question, context)
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
        return QuestionResult(
            question=question, answer=answer, metrics=metrics,
            chunks=chunks, scores=scores,
        )


# ---------------------------------------------------------------------------
# E03 – RAG with Sentence Evidence
# ---------------------------------------------------------------------------

class E03_SentenceEvidence(BaseExperiment):
    CONFIG = {
        "experiment_id": "E03",
        "system_type":   "RAG with Sentence Evidence",
        "llm":           LLM_MODEL,
        "embedding_model": EMBED_MODEL,
        "vector_db":     "ChromaDB",
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    def setup(self) -> None:
        self.vector_store = self._build_vector_store()

    def run_question(self, question: str, ground_truth: str) -> QuestionResult:
        hits = self.vector_store.query(question, n_results=TOP_K)
        chunks = [h["text"] for h in hits]
        scores = [h["score"] for h in hits]

        evidence = extract_sentence_evidence(question, chunks, self.embedder, top_n=5)
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
        return QuestionResult(
            question=question, answer=answer, metrics=metrics,
            chunks=chunks, scores=scores,
        )


# ---------------------------------------------------------------------------
# E04 – RAG with Uncertainty Handling
# ---------------------------------------------------------------------------

class E04_UncertaintyRAG(BaseExperiment):
    CONFIG = {
        "experiment_id": "E04",
        "system_type":   "RAG with Uncertainty Handling",
        "llm":           LLM_MODEL,
        "embedding_model": EMBED_MODEL,
        "vector_db":     "ChromaDB",
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    HIGH_CONF  = 0.60
    LOW_CONF   = 0.40

    def setup(self) -> None:
        self.vector_store = self._build_vector_store()

    def _confidence_label(self, scores: list[float]) -> str:
        if not scores:
            return "Low"
        top = max(scores)
        if top >= self.HIGH_CONF:
            return "High"
        if top >= self.LOW_CONF:
            return "Medium"
        return "Low"

    def run_question(self, question: str, ground_truth: str) -> QuestionResult:
        hits = self.vector_store.query(question, n_results=TOP_K)
        chunks = [h["text"] for h in hits]
        scores = [h["score"] for h in hits]
        context = "\n\n".join(chunks)
        confidence = self._confidence_label(scores)

        answer = self.llm.answer_uncertainty(question, context, confidence)
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
        return QuestionResult(
            question=question, answer=answer, metrics=metrics,
            chunks=chunks, scores=scores,
        )


# ---------------------------------------------------------------------------
# E05 – Final Combined RAG
# ---------------------------------------------------------------------------

class E05_CombinedRAG(BaseExperiment):
    CONFIG = {
        "experiment_id": "E05",
        "system_type":   "Final Combined RAG System",
        "llm":           LLM_MODEL,
        "embedding_model": EMBED_MODEL,
        "vector_db":     "ChromaDB",
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    HIGH_CONF = 0.60
    LOW_CONF  = 0.40

    def setup(self) -> None:
        self.vector_store = self._build_vector_store()

    def _expand_query(self, question: str) -> str:
        """Simple query expansion: append medical synonyms for key terms."""
        expansions = {
            "symptoms":   "signs manifestations",
            "treatment":  "therapy management",
            "cause":      "etiology risk factors",
            "prevent":    "prevention reduce risk",
            "diagnose":   "diagnosis detection screening",
        }
        extra: list[str] = []
        q_lower = question.lower()
        for key, synonyms in expansions.items():
            if key in q_lower:
                extra.append(synonyms)
        return question + (" " + " ".join(extra) if extra else "")

    def _confidence_label(self, scores: list[float]) -> str:
        if not scores:
            return "Low"
        top = max(scores)
        if top >= self.HIGH_CONF:
            return "High"
        if top >= self.LOW_CONF:
            return "Medium"
        return "Low"

    def run_question(self, question: str, ground_truth: str) -> QuestionResult:
        # Query expansion + broader retrieval
        expanded = self._expand_query(question)
        hits = self.vector_store.query(expanded, n_results=TOP_K + 2)
        chunks = [h["text"] for h in hits]
        scores = [h["score"] for h in hits]

        # Sentence-level evidence extraction (more sentences than E03)
        evidence = extract_sentence_evidence(question, chunks, self.embedder, top_n=7)

        # Rank evidence lines by score label
        confidence = self._confidence_label(scores)
        ranked_evidence = f"[Confidence: {confidence}]\n{evidence}"

        answer = self.llm.answer_combined(question, ranked_evidence, confidence)
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
        return QuestionResult(
            question=question, answer=answer, metrics=metrics,
            chunks=chunks, scores=scores,
        )
