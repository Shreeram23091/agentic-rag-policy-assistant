__all__ = ["Agent", "AgentResponse"]


def __getattr__(name):
    if name in ("Agent", "AgentResponse"):
        from app.agent.agent import Agent, AgentResponse
        return Agent if name == "Agent" else AgentResponse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
