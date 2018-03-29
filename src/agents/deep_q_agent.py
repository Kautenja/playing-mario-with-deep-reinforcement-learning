"""An implementation of Deep Q-Learning."""
from .agent import Agent


class DeepQAgent(Agent):
    """An implementation of Deep Q-Learning."""

    def __init__(self) -> None:
        """Create a new Deep Q reinforcement agent."""
        pass


# explicitly define the outward facing API of this module
__all__ = ['DeepQAgent']
