"""
app/api/routes.py — FastAPI route definitions.

Endpoints:
  POST /ask        — Main agent endpoint
  GET  /health     — Health + readiness check
  DELETE /session/{session_id} — Clear a session
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.agent.agent import Agent
from app.api.dependencies import get_agent, get_memory, get_retriever
from app.api.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse
from app.memory.session_memory import SessionMemory
from app.rag.retriever import RAGRetriever

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── POST /ask ─────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "RAG not ready"},
    },
    summary="Ask the AI agent a question",
    description=(
        "Send a natural-language query. The agent decides whether to answer directly "
        "or retrieve information from internal documents. Optionally provide a session_id "
        "for multi-turn conversation memory."
    ),
)
async def ask(
    request: AskRequest,
    agent: Agent = Depends(get_agent),
    memory: SessionMemory = Depends(get_memory),
    retriever: RAGRetriever = Depends(get_retriever),
) -> AskResponse:
    log = logger.bind(query=request.query[:80], session_id=request.session_id)
    log.info("Received /ask request")

    # Load conversation history if a session_id is provided
    history = []
    if request.session_id:
        history = memory.get_history(request.session_id)

    try:
        result = await agent.run(user_query=request.query, history=history)
    except Exception as exc:
        log.exception("Agent error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent encountered an error: {exc}",
        ) from exc

    # Persist exchange to session memory
    if request.session_id:
        memory.add_exchange(
            session_id=request.session_id,
            user_msg=request.query,
            assistant_msg=result.answer,
        )

    log.info(
        "Request complete",
        used_rag=result.used_rag,
        sources=result.sources,
    )

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        session_id=request.session_id,
        used_rag=result.used_rag,
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
)
async def health(
    retriever: RAGRetriever = Depends(get_retriever),
    memory: SessionMemory = Depends(get_memory),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        rag_ready=retriever.is_ready,
        active_sessions=memory.active_session_count,
    )


# ── DELETE /session/{session_id} ──────────────────────────────────────────────

# ── DELETE /session/{session_id} ──────────────────────────────────────────────

@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,  # Changed from 204 to 200 to allow a response message
    summary="Clear conversation history for a session",
)
async def clear_session(
    session_id: str,
    memory: SessionMemory = Depends(get_memory),
):
    """
    Clears the conversation history for the specified session ID.
    Returns a success message upon completion.
    """
    memory.clear_session(session_id)
    logger.info("Session cleared", session_id=session_id)
    
    return {"status": "success", "message": f"History for session {session_id} has been cleared."}