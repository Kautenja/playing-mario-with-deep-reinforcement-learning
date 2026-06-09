"""Legacy Keras models for value function estimation in deep RL."""

__all__ = ["build_deep_q_model", "build_dueling_deep_q_model"]


def __getattr__(name):
    """Lazily import legacy model builders and their Keras dependency."""
    if name == "build_deep_q_model":
        from .deep_q_model import build_deep_q_model

        return build_deep_q_model
    if name == "build_dueling_deep_q_model":
        from .dueling_deep_q_model import build_dueling_deep_q_model

        return build_dueling_deep_q_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
