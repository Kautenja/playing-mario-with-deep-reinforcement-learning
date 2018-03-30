"""Various deep learning models for value function estimation in deep RL."""
from .deep_mind_model import build_deep_mind_model


# explicitly define the outward facing API for this package
__all__ = ['build_deep_mind_model']
