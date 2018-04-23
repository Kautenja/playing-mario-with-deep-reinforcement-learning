"""A package with implementations of deep reinforcement agents."""
from .random_agent import RandomAgent
from .deep_q_agent import DeepQAgent


# explicitly define the outward facing API of this package.
__all__ = [
    RandomAgent.__class__,
    DeepQAgent.__class__,
]
