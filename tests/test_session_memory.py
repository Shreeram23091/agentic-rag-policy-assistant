"""tests/test_session_memory.py — Unit tests for SessionMemory."""
import time
import pytest
from unittest.mock import patch

from app.memory.session_memory import SessionMemory


@pytest.fixture
def memory():
    return SessionMemory()


def test_empty_history_for_new_session(memory):
    assert memory.get_history("new-session") == []


def test_add_and_retrieve_exchange(memory):
    memory.add_exchange("s1", "Hello", "Hi there!")
    history = memory.get_history("s1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there!"}


def test_multiple_exchanges(memory):
    memory.add_exchange("s1", "Q1", "A1")
    memory.add_exchange("s1", "Q2", "A2")
    history = memory.get_history("s1")
    assert len(history) == 4


def test_clear_session(memory):
    memory.add_exchange("s2", "Q", "A")
    memory.clear_session("s2")
    assert memory.get_history("s2") == []


def test_session_cap(memory):
    """History should be capped at max_messages."""
    # Add more than max_history_turns exchanges
    for i in range(20):
        memory.add_exchange("s3", f"Q{i}", f"A{i}")
    history = memory.get_history("s3")
    from app.config import get_settings
    max_msgs = get_settings().max_history_turns * 2
    assert len(history) <= max_msgs


def test_active_session_count(memory):
    assert memory.active_session_count == 0
    memory.add_exchange("s1", "q", "a")
    memory.add_exchange("s2", "q", "a")
    assert memory.active_session_count == 2


def test_session_expiry(memory):
    """Expired sessions should return empty history."""
    memory.add_exchange("expire-me", "q", "a")
    # Manually expire by backdating last_accessed
    with memory._lock:
        memory._sessions["expire-me"].last_accessed = time.time() - 99999
    assert memory.get_history("expire-me") == []
