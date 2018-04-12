"""A package with implementations of deep reinforcement agents."""
from .random_agent import RandomAgent
from .deep_q_agent import DeepQAgent
from .a3c_agent import A3C_Agent


# explicitly define the outward facing API of this package.
__all__ = [
    RandomAgent.__class__,
    DeepQAgent.__class__,
    A3C_Agent.__class__,
]
