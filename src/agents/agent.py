"""An abstract base class for deep reinforcement agents."""


class Agent(object):
    """An abstract base class for building deep reinforcement agents."""

    def __init__(self) -> None:
        """Create a new abstract deep reinforcement agent."""
        pass


# explicitly define the outward facing API of this module
__all__ = ['Agent']
