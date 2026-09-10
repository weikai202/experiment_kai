"""Deterministic generation-pinned semantic retrieval."""


def __getattr__(name):
    if name == "RetrievalService":
        from .service import RetrievalService
        return RetrievalService
    raise AttributeError(name)


__all__ = ["RetrievalService"]
