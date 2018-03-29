"""An implementation of Double Deep Q-Learning."""
from .agent import Agent


class DoubleDeepQAgent(Agent):
    """An implementation of Double Deep Q-Learning."""

    def __init__(self) -> None:
        """Create a new Double Deep Q reinforcement agent."""
        pass


# explicitly define the outward facing API of this module
__all__ = ['DoubleDeepQAgent']
