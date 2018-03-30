"""An implementation of random search deep reinforcement."""
from .agent import Agent


class RandomAgent(Agent):
    """An implementation of random search deep reinforcement."""

    def __init__(self) -> None:
        """Create a new random search reinforcement agent."""
        pass


# explicitly define the outward facing API of this module
__all__ = ['RandomAgent']
