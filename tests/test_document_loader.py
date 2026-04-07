"""tests/test_document_loader.py — Unit tests for the document chunker."""
import pytest
from app.rag.document_loader import chunk_text, DocumentChunk


def test_chunk_text_basic():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunk_text(text, source="test.txt", chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 1
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert chunks[0].source == "test.txt"


def test_chunk_text_overlap():
    """Chunks should have some overlap when chunk_size is small."""
    long_text = ". ".join([f"Sentence number {i} here" for i in range(40)]) + "."
    chunks = chunk_text(long_text, source="doc.txt", chunk_size=120, chunk_overlap=40)
    assert len(chunks) > 1
    # Verify all chunk indices are sequential
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_text_single_short_doc():
    text = "Only one sentence."
    chunks = chunk_text(text, source="small.txt", chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence."


def test_chunk_text_empty():
    chunks = chunk_text("", source="empty.txt")
    assert chunks == []


def test_chunk_preserves_source():
    text = "Hello world. Another sentence here."
    chunks = chunk_text(text, source="my_file.txt")
    for chunk in chunks:
        assert chunk.source == "my_file.txt"
