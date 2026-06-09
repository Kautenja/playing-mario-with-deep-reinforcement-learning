"""Base components for the legacy project."""

__all__ = ["AnnealingVariable", "PrioritizedReplayQueue", "ReplayQueue"]


def __getattr__(name):
    """Lazily import legacy base helpers."""
    if name == "AnnealingVariable":
        from .annealing_variable import AnnealingVariable

        return AnnealingVariable
    if name == "PrioritizedReplayQueue":
        from .prioritized_replay_queue import PrioritizedReplayQueue

        return PrioritizedReplayQueue
    if name == "ReplayQueue":
        from .replay_queue import ReplayQueue

        return ReplayQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
