"""A package with legacy reinforcement agents."""

__all__ = ["RandomAgent", "DeepQAgent"]


def __getattr__(name):
    """Lazily import legacy agents and their optional framework dependencies."""
    if name == "RandomAgent":
        from .random_agent import RandomAgent

        return RandomAgent
    if name == "DeepQAgent":
        from .deep_q_agent import DeepQAgent

        return DeepQAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
