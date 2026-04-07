"""
app/rag/retriever.py — High-level RAG retriever.

Orchestrates: query embedding → FAISS search → chunk deduplication → context assembly.
Also exposes an initialise() method that builds/loads the FAISS index on startup.
"""
from __future__ import annotations

import structlog

from app.config import get_settings
from app.rag.document_loader import load_and_chunk_all
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import FAISSVectorStore

logger = structlog.get_logger(__name__)


class RAGRetriever:
    """
    Main retriever class.

    Usage:
        retriever = RAGRetriever()
        await retriever.initialise()
        results = await retriever.retrieve("What is the leave policy?")
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._vector_store = FAISSVectorStore()
        self._embedding_service = EmbeddingService()
        self._ready = False

    async def initialise(self, force_rebuild: bool = False) -> None:
        """
        Load existing FAISS index from disk or build it from documents.
        Set force_rebuild=True to regenerate embeddings even if an index exists.
        """
        if not force_rebuild and self._vector_store.load(self._settings.faiss_index_path):
            logger.info("RAG retriever ready (loaded from disk)")
            self._ready = True
            return

        logger.info("Building FAISS index from documents…")
        chunks = load_and_chunk_all(
            documents_path=self._settings.documents_path,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )

        if not chunks:
            logger.error("No document chunks found; RAG will be unavailable")
            return

        texts = [c.text for c in chunks]
        logger.info("Generating embeddings", num_chunks=len(texts))
        embeddings = await self._embedding_service.embed_texts(texts)

        self._vector_store.build(chunks, embeddings)
        self._vector_store.save(self._settings.faiss_index_path)
        self._ready = True
        logger.info("FAISS index built and saved", num_vectors=len(chunks))

    async def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the top-k most relevant document chunks for a query.

        Returns a list of dicts with keys: text, source, score, chunk_index.
        """
        if not self._ready or not self._vector_store.is_ready:
            logger.warning("Retriever not ready; returning empty results")
            return []

        query_embedding = await self._embedding_service.embed_query(query)
        results = self._vector_store.search(query_embedding, top_k=self._settings.top_k_results)

        output = []
        seen_texts: set[str] = set()
        for chunk, score in results:
            # Deduplicate near-identical chunks
            if chunk.text[:100] in seen_texts:
                continue
            seen_texts.add(chunk.text[:100])
            output.append({
                "text": chunk.text,
                "source": chunk.source,
                "score": round(score, 4),
                "chunk_index": chunk.chunk_index,
            })

        logger.debug("Retrieval complete", query=query[:60], num_results=len(output))
        return output

    @property
    def is_ready(self) -> bool:
        return self._ready
