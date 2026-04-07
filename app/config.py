"""
app/config.py — Centralised settings via Pydantic BaseSettings.
All values are loaded from environment variables (or .env file).
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI / Azure OpenAI ────────────────────────────────────────────────
    openai_api_type: Literal["openai", "azure"] = "openai"

    # Standard OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o"
    openai_embed_model: str = "text-embedding-3-small"

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embed_deployment: str = "text-embedding-3-small"

    # ── RAG / Vector Store ───────────────────────────────────────────────────
    faiss_index_path: str = "./data/faiss_index"
    documents_path: str = "./docs/sample_documents"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 4

    # ── Session / Memory ─────────────────────────────────────────────────────
    session_ttl_minutes: int = 60
    max_history_turns: int = 10

    # ── API Config ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: Literal["development", "staging", "production"] = "development"
    api_secret_key: str = Field(default="change-me-in-production")
    allowed_origins: str = "http://localhost:3000"

    # ── Azure Monitor ────────────────────────────────────────────────────────
    applicationinsights_connection_string: str = ""

    @property
    def chat_model(self) -> str:
        return (
            self.azure_openai_chat_deployment
            if self.openai_api_type == "azure"
            else self.openai_chat_model
        )

    @property
    def embed_model(self) -> str:
        return (
            self.azure_openai_embed_deployment
            if self.openai_api_type == "azure"
            else self.openai_embed_model
        )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_azure(self) -> bool:
        return self.openai_api_type == "azure"


@lru_cache
def get_settings() -> Settings:
    return Settings()
