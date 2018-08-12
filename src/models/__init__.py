"""Deep learning models for value function estimation in deep RL."""
from .deep_q_model import build_deep_q_model
from .dueling_deep_q_model import build_dueling_deep_q_model


# explicitly define the outward facing API for this package
__all__ = [
    build_deep_q_model.__name__,
    build_dueling_deep_q_model.__name__,
]
