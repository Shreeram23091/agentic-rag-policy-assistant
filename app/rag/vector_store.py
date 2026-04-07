"""
app/rag/vector_store.py — FAISS-backed vector store.

Handles:
  - Building and persisting a FAISS index from document chunks
  - Loading an existing index from disk
  - Similarity search returning top-k chunks with scores
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import structlog

from app.rag.document_loader import DocumentChunk

logger = structlog.get_logger(__name__)

# Lazy imports for FAISS so the app can start without it (will raise on first use)
import faiss
FAISS_AVAILABLE = True


class FAISSVectorStore:
    """
    Wraps a FAISS IndexFlatIP (inner product / cosine similarity) index.
    Embeddings are L2-normalised before indexing so inner product == cosine.
    """

    METADATA_FILE = "metadata.json"
    INDEX_FILE = "index.faiss"

    def __init__(self) -> None:
        self._index: "faiss.IndexFlatIP | None" = None  # type: ignore[name-defined]
        self._chunks: list[DocumentChunk] = []

    # ── Build & Persist ──────────────────────────────────────────────────────

    def build(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Build the FAISS index from chunks and their pre-computed embeddings."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu")

        dim = len(embeddings[0])
        self._index = faiss.IndexFlatIP(dim)

        vectors = np.array(embeddings, dtype="float32")
        # L2-normalise for cosine similarity
        faiss.normalize_L2(vectors)
        self._index.add(vectors)

        self._chunks = chunks
        logger.info("FAISS index built", num_vectors=self._index.ntotal, dim=dim)

    def save(self, index_path: str) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        if self._index is None:
            raise RuntimeError("Index not built yet; call build() first.")

        path = Path(index_path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / self.INDEX_FILE))

        metadata = [
            {
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        (path / self.METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Vector store saved", path=str(path))

    def load(self, index_path: str) -> bool:
        """Load index from disk. Returns True if successful, False if not found."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is not installed.")

        path = Path(index_path)
        index_file = path / self.INDEX_FILE
        meta_file = path / self.METADATA_FILE

        if not index_file.exists() or not meta_file.exists():
            logger.warning("FAISS index not found on disk", path=str(path))
            return False

        self._index = faiss.read_index(str(index_file))
        raw_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self._chunks = [
            DocumentChunk(
                text=m["text"],
                source=m["source"],
                chunk_index=m["chunk_index"],
                metadata=m.get("metadata", {}),
            )
            for m in raw_meta
        ]
        logger.info("Vector store loaded", num_vectors=self._index.ntotal)
        return True

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Return top-k (chunk, score) pairs ranked by cosine similarity.
        Score is in [−1, 1]; higher is more similar.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)

        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))

        results: list[tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS uses -1 for missing
                continue
            results.append((self._chunks[idx], float(score)))

        return results

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0
