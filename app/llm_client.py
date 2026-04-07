"""
app/llm_client.py — Factory that returns the correct OpenAI client
(standard OpenAI or Azure OpenAI) based on configuration.
"""
from functools import lru_cache

from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.config import Settings, get_settings


@lru_cache
def get_chat_client(settings: Settings | None = None) -> AsyncOpenAI | AsyncAzureOpenAI:
    """Return an async OpenAI (or Azure OpenAI) client for chat completions."""
    cfg = settings or get_settings()
    if cfg.is_azure:
        return AsyncAzureOpenAI(
            api_key=cfg.azure_openai_api_key,
            azure_endpoint=cfg.azure_openai_endpoint,
            api_version=cfg.azure_openai_api_version,
        )
    return AsyncOpenAI(api_key=cfg.openai_api_key)


@lru_cache
def get_embed_client(settings: Settings | None = None) -> AsyncOpenAI | AsyncAzureOpenAI:
    """Return an async OpenAI (or Azure OpenAI) client for embeddings."""
    # Same client class; separated for potential future divergence
    return get_chat_client(settings)
