"""Shared word-based text chunker with configurable overlap."""
from __future__ import annotations

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]
    step, chunks, start = chunk_size - chunk_overlap, [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Return a flat list of chunk dicts from a list of knowledge-base documents."""
    out: list[dict] = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc["cleaned_text"], chunk_size, chunk_overlap)):
            out.append({
                "chunk_id":     f"{doc['document_id']}_C{i+1:02d}",
                "document_id":  doc["document_id"],
                "text":         text,
                "title":        doc.get("title", ""),
                "source":       doc.get("source", ""),
                "medical_topic":doc.get("medical_topic", ""),
                "source_link":  doc.get("source_link", ""),
            })
    return out
