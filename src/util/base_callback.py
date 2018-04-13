"""A reward tracking callback for the command line."""


class BaseCallback(object):
    """A reward tracking callback for the command line."""

    def __init__(self) -> None:
        """Create a new Shell Callback method."""
        # setup caches for metrics
        self.scores = []
        self.losses = []

    def __call__(self, score: float, loss: float) -> None:
        """
        Update the callback with the new score (from a finished episode).

        Args:
            agent: the agent reporting the loss
            score: the score to log
            loss: the loss from training the network to log

        Returns:
            None

        """
        # append the score to the list
        self.scores.append(score)
        self.losses.append(loss)


# explicitly define the outward facing API of this module
__all__ = [BaseCallback.__name__]
