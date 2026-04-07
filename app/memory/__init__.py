__all__ = ["SessionMemory"]


def __getattr__(name):
    if name == "SessionMemory":
        from app.memory.session_memory import SessionMemory
        return SessionMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
