"""
app/api/schemas.py — Pydantic models for request/response validation.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question or message.",
        examples=["How many days of annual leave do I get?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional session ID for conversation continuity. "
                    "If omitted, a stateless single-turn response is returned.",
        examples=["user-abc-123"],
    )


# ── Response ──────────────────────────────────────────────────────────────────

class AskResponse(BaseModel):
    answer: str = Field(description="The agent's answer to the query.")
    sources: list[str] = Field(
        default_factory=list,
        description="List of document filenames referenced in the answer.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Echo of the session_id (if provided).",
    )
    used_rag: bool = Field(
        default=False,
        description="Whether RAG retrieval was used to generate this answer.",
    )


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool
    active_sessions: int
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
