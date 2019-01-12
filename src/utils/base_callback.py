"""A reward tracking callback for the command line."""


class BaseCallback(object):
    """A reward tracking callback for the command line."""

    def __init__(self, weights_file_name: str, update_every: int=100) -> None:
        """
        Initialize a new base callback.

        Args:
            weights_file_name: the name of the file to save model weights to
            update_every: the number of episodes to see before saving weights

        Returns:
            None

        """
        # setup caches for metrics
        self.weights_file_name = weights_file_name
        self.update_every = update_every
        self._episodes = 0
        self.scores = []
        self.losses = []

    def __repr__(self) -> str:
        """Return an executable string representation of this object."""
        return '{}(weights_file_name={}, update_every={})'.format(
            self.__class__.__name__,
            repr(self.weights_file_name),
            self.update_every,
        )

    def __call__(self, agent, score: float, loss: float) -> None:
        """
        Update the callback with the new score (from a finished episode).

        Args:
            agent: the agent producing the score and loss
            score: the score to log
            loss: the loss from training the network to log

        Returns:
            None

        """
        # increment the episode counter
        self._episodes += 1
        # append the score to the list
        self.scores.append(score)
        self.losses.append(loss)
        # save the weights
        if self._episodes % self.update_every == 0:
            agent.model.save_weights(self.weights_file_name)


# explicitly define the outward facing API of this module
__all__ = [BaseCallback.__name__]
