"""
app/rag/document_loader.py — Loads, chunks, and prepares documents for embedding.

Supports .txt and .pdf files. Chunks are split by sentence-aware sliding window
with configurable size and overlap.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of text extracted from a source document."""
    text: str
    source: str          # filename without path
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pypdf not installed; skipping PDF", file=str(path))
        return ""


def load_documents(documents_path: str) -> list[tuple[str, str]]:
    """
    Load all .txt and .pdf files from a directory.
    Returns list of (filename, full_text) tuples.
    """
    docs_dir = Path(documents_path)
    if not docs_dir.exists():
        logger.error("Documents directory not found", path=documents_path)
        return []

    documents: list[tuple[str, str]] = []
    for file_path in sorted(docs_dir.iterdir()):
        if file_path.suffix.lower() == ".txt":
            text = _load_txt(file_path)
        elif file_path.suffix.lower() == ".pdf":
            text = _load_pdf(file_path)
        else:
            continue

        if text.strip():
            documents.append((file_path.name, text))
            logger.info("Loaded document", file=file_path.name, chars=len(text))

    return documents


def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter on '. ', '! ', '? ' boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Sliding-window chunker that respects approximate token/character boundaries.
    chunk_size and chunk_overlap are measured in characters.
    """
    sentences = _split_into_sentences(text)
    chunks: list[DocumentChunk] = []
    current_chunk: list[str] = []
    current_len = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed chunk_size, flush
        if current_len + sentence_len > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(
                DocumentChunk(
                    text=chunk_text_str,
                    source=source,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            # Overlap: keep the last ~chunk_overlap chars worth of sentences
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += sentence_len

    # Flush remaining
    if current_chunk:
        chunks.append(
            DocumentChunk(
                text=" ".join(current_chunk),
                source=source,
                chunk_index=chunk_index,
            )
        )

    return chunks


def load_and_chunk_all(
    documents_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Load all documents from a directory and return a flat list of chunks."""
    raw_docs = load_documents(documents_path)
    all_chunks: list[DocumentChunk] = []
    for filename, text in raw_docs:
        chunks = chunk_text(text, source=filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
        logger.info("Chunked document", file=filename, num_chunks=len(chunks))
    logger.info("Total chunks prepared", total=len(all_chunks))
    return all_chunks
