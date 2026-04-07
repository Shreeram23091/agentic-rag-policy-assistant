# Lazy imports — consumers should import directly from submodules
__all__ = ["RAGRetriever"]


def __getattr__(name):
    if name == "RAGRetriever":
        from app.rag.retriever import RAGRetriever
        return RAGRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
