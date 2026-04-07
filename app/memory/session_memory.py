"""
app/memory/session_memory.py — In-process session-based conversation memory.

Stores per-session chat history with TTL-based expiration.
In production, replace with Redis for multi-instance deployments.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TypedDict

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)


class MessageDict(TypedDict):
    role: str   # "user" | "assistant" | "system"
    content: str


@dataclass
class Session:
    session_id: str
    messages: list[MessageDict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed = time.time()

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.last_accessed) > ttl_seconds


class SessionMemory:
    """
    Thread-safe, in-memory session store.

    Each session maintains a list of chat messages (role + content),
    capped at max_turns * 2 messages (each turn = 1 user + 1 assistant).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()
        cfg = get_settings()
        self._ttl_seconds = cfg.session_ttl_minutes * 60
        self._max_messages = cfg.max_history_turns * 2  # user + assistant per turn

    # ── Public API ───────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> list[MessageDict]:
        """Return conversation history for a session (empty list if new/expired)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired(self._ttl_seconds):
                if session:
                    logger.info("Session expired", session_id=session_id)
                    del self._sessions[session_id]
                return []
            session.touch()
            return list(session.messages)

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Append a user/assistant exchange to a session, respecting max cap."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired(self._ttl_seconds):
                session = Session(session_id=session_id)
                self._sessions[session_id] = session
                logger.info("New session created", session_id=session_id)

            session.messages.append({"role": "user", "content": user_msg})
            session.messages.append({"role": "assistant", "content": assistant_msg})

            # Trim to max window (keep most recent)
            if len(session.messages) > self._max_messages:
                session.messages = session.messages[-self._max_messages :]

            session.touch()

    def clear_session(self, session_id: str) -> None:
        """Explicitly clear a session (e.g., user logout)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def purge_expired(self) -> int:
        """Remove all expired sessions. Returns count of removed sessions."""
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if s.is_expired(self._ttl_seconds)
            ]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                logger.info("Purged expired sessions", count=len(expired))
            return len(expired)

    @property
    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
