"""
tests/test_api.py — Integration tests for FastAPI endpoints.

Uses FastAPI dependency_overrides to inject mocks cleanly,
with no real OpenAI calls and no lifespan complications.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from app.agent.agent import Agent, AgentResponse
from app.memory.session_memory import SessionMemory
from app.rag.retriever import RAGRetriever


# ── Shared mock fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def mock_retriever():
    r = MagicMock(spec=RAGRetriever)
    r.is_ready = True
    r.retrieve = AsyncMock(return_value=[
        {"text": "You get 18 days annual leave.", "source": "leave_policy.txt", "score": 0.92, "chunk_index": 0}
    ])
    return r


@pytest.fixture
def mock_memory():
    return SessionMemory()


@pytest.fixture
def mock_agent():
    a = MagicMock(spec=Agent)
    a.run = AsyncMock(return_value=AgentResponse(
        answer="You are entitled to 18 days of annual leave.",
        sources=["leave_policy.txt"],
        used_rag=True,
    ))
    return a


@pytest_asyncio.fixture
async def client(mock_retriever, mock_memory, mock_agent):
    """Build test client using dependency_overrides — no lifespan needed."""
    from app.main import create_app
    from app.api import dependencies

    app = create_app()
    app.dependency_overrides[dependencies.get_retriever] = lambda: mock_retriever
    app.dependency_overrides[dependencies.get_memory] = lambda: mock_memory
    app.dependency_overrides[dependencies.get_agent] = lambda: mock_agent

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ask_basic(client):
    response = await client.post("/ask", json={"query": "How many leave days do I get?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "You are entitled to 18 days of annual leave."
    assert data["sources"] == ["leave_policy.txt"]
    assert data["used_rag"] is True
    assert data["session_id"] is None


@pytest.mark.asyncio
async def test_ask_with_session(client, mock_memory):
    response = await client.post("/ask", json={
        "query": "What is the leave policy?",
        "session_id": "test-session-001",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-001"
    history = mock_memory.get_history("test-session-001")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_ask_empty_query_rejected(client):
    response = await client.post("/ask", json={"query": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ask_missing_query_rejected(client):
    response = await client.post("/ask", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint_ready(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["rag_ready"] is True
    assert "active_sessions" in data
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_clear_session(client, mock_memory):
    mock_memory.add_exchange("clear-me", "q", "a")
    assert len(mock_memory.get_history("clear-me")) == 2
    response = await client.delete("/session/clear-me")
    assert response.status_code == 204
    assert mock_memory.get_history("clear-me") == []


@pytest.mark.asyncio
async def test_root_returns_json(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


@pytest.mark.asyncio
async def test_multiturn_history_grows(client, mock_memory):
    """Three requests with the same session_id should accumulate 6 messages."""
    for i in range(3):
        await client.post("/ask", json={"query": f"Question {i}", "session_id": "multi-turn"})
    history = mock_memory.get_history("multi-turn")
    assert len(history) == 6


@pytest.mark.asyncio
async def test_agent_receives_prior_history(client, mock_agent, mock_memory):
    """Second call in a session should pass the first exchange as history."""
    mock_memory.add_exchange("hist-session", "First question", "First answer")
    await client.post("/ask", json={"query": "Follow-up", "session_id": "hist-session"})
    call_kwargs = mock_agent.run.call_args
    # history is passed as keyword argument
    history_arg = call_kwargs.kwargs.get("history", [])
    assert len(history_arg) >= 2
