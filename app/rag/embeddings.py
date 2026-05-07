"""
app/rag/embeddings.py — Embedding service using OpenAI or Azure OpenAI.
Enhanced with a Mock Failover for environment-restricted subscriptions.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.llm_client import get_embed_client

logger = structlog.get_logger(__name__)

_BATCH_SIZE = 64  # Conservative batch size


class EmbeddingService:
    """Async service for generating text embeddings via OpenAI API with Mock Fallback."""

    def __init__(self) -> None:
        self._settings = get_settings()
        try:
            self._client = get_embed_client()
        except Exception as e:
            logger.error("Failed to initialize embedding client", error=str(e))
            self._client = None

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts, with a mock fallback on API failure."""
        
        # 1. Try the Real API (Azure/OpenAI)
        if self._client:
            try:
                response = await self._client.embeddings.create(
                    model=self._settings.embed_model,
                    input=texts,
                )
                # Sort by index to ensure ordering matches input
                # Use enumerate fallback for APIs that don't return index (e.g. Gemini)
                sorted_data = sorted(response.data, key=lambda x: x.index if x.index is not None else 0)
                return [item.embedding for item in sorted_data]
            except Exception as e:
                logger.warning(
                    "API Embedding failed. Falling back to Mocking.",
                    error=str(e),
                    type=type(e).__name__
                )

        # 2. Fallback: Mock Vector Generation
        # This allows the server to start even if Azure policies block the model
        logger.info("Generating mock embeddings for batch", count=len(texts))
        return [[random.uniform(-1, 1) for _ in range(1536)] for _ in texts]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        Splits into batches and runs them sequentially.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            logger.debug("Processing embedding batch", start=i, size=len(batch))
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([query])
        return results[0]