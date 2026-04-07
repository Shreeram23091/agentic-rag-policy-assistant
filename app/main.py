"""
app/main.py — FastAPI application factory.

Lifespan:
  startup  → initialise RAGRetriever (build/load FAISS index), wire up singletons
  shutdown → log active session count

Middleware:
  - CORS
  - Request ID injection for tracing
  - Structured logging middleware
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.agent import Agent
from app.api.dependencies import set_singletons
from app.api.routes import router
from app.config import get_settings
from app.memory.session_memory import SessionMemory
from app.rag.retriever import RAGRetriever

logger = structlog.get_logger(__name__)


# ── Structured logging setup ─────────────────────────────────────────────────

def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if get_settings().environment == "development"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    cfg = get_settings()
    logger.info("Starting ACME RAG Agent", environment=cfg.environment)

    # Initialise singletons
    retriever = RAGRetriever()
    memory = SessionMemory()

    await retriever.initialise()

    agent = Agent(retrieve_fn=retriever.retrieve)
    set_singletons(retriever=retriever, memory=memory, agent=agent)

    logger.info("Application ready", rag_ready=retriever.is_ready)
    yield

    logger.info(
        "Shutting down",
        active_sessions=memory.active_session_count,
    )


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="ACME RAG Agent API",
        description=(
            "An AI agent that answers questions about internal company documents "
            "using Retrieval-Augmented Generation (RAG). "
            "Powered by OpenAI / Azure OpenAI + FAISS."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID + latency middleware
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()

        response: Response = await call_next(request)

        elapsed = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "HTTP",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=elapsed,
        )
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.clear_contextvars()
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred.", "error_type": type(exc).__name__},
        )

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({"message": "ACME RAG Agent API", "docs": "/docs"})

    app.include_router(router)
    return app


app = create_app()
