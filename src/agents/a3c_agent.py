"""An implementation of Asynchronous Advantage Actor-Critic (A3C)."""
from .agent import Agent


class A3C_Agent(Agent):
    """An implementation of Asynchronous Advantage Actor-Critic (A3C)."""

    def __init__(self) -> None:
        """Create a new A3C deep reinforcement agent."""
        pass


# explicitly define the outward facing API of this module
__all__ = ['A3C_Agent']
