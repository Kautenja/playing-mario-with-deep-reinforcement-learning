"""Various deep learning models for value function estimation in deep RL."""
from .deep_mind_model import build_deep_mind_model
from .a3c_model import build_a3c_model


# explicitly define the outward facing API for this package
__all__ = [
    build_deep_mind_model.__name__,
    build_a3c_model.__name__
]
