"""
app/api/dependencies.py — FastAPI dependency injection for shared singletons.

RAGRetriever, SessionMemory, and Agent are created once at startup
and injected into route handlers via Depends().
"""
from __future__ import annotations

from functools import lru_cache

from app.agent.agent import Agent
from app.memory.session_memory import SessionMemory
from app.rag.retriever import RAGRetriever

# Module-level singletons — set during app lifespan startup
_retriever: RAGRetriever | None = None
_memory: SessionMemory | None = None
_agent: Agent | None = None


def set_singletons(retriever: RAGRetriever, memory: SessionMemory, agent: Agent) -> None:
    global _retriever, _memory, _agent
    _retriever = retriever
    _memory = memory
    _agent = agent


def get_retriever() -> RAGRetriever:
    if _retriever is None:
        raise RuntimeError("RAGRetriever not initialised")
    return _retriever


def get_memory() -> SessionMemory:
    if _memory is None:
        raise RuntimeError("SessionMemory not initialised")
    return _memory


def get_agent() -> Agent:
    if _agent is None:
        raise RuntimeError("Agent not initialised")
    return _agent
