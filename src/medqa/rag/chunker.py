"""Word-based text chunker with overlap."""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum number of words per chunk.
        chunk_overlap: Number of words shared between adjacent chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [text]

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """Chunk a list of knowledge-base documents.

    Each document dict must have at least: document_id, cleaned_text.
    Returns a flat list of chunk dicts with keys:
        chunk_id, document_id, text, title, source, medical_topic, source_link.
    """
    chunks: list[dict] = []
    for doc in documents:
        doc_chunks = chunk_text(doc["cleaned_text"], chunk_size, chunk_overlap)
        for i, chunk_text_str in enumerate(doc_chunks):
            chunks.append(
                {
                    "chunk_id": f"{doc['document_id']}_C{i + 1:02d}",
                    "document_id": doc["document_id"],
                    "text": chunk_text_str,
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "medical_topic": doc.get("medical_topic", ""),
                    "source_link": doc.get("source_link", ""),
                }
            )
    return chunks
