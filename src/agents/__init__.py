"""A package with implementations of deep reinforcement agents."""
from .agent import Agent
from .random_agent import RandomAgent
from .deep_q_agent import DeepQAgent


# explicitly define the outward facing API of this package.
__all__ = [
    Agent.__name__,
    RandomAgent.__class__,
    DeepQAgent.__class__,
]
