"""A rich reward tracking callback for Jupyter notebooks."""
from matplotlib import pyplot as plt
from IPython import display


class JupyterCallback(object):
    """A rich reward tracking callback for Jupyter notebooks."""

    def __init__(self) -> None:
        """Create a new Jupyter Callback method."""
        # setup instance members
        self.scores = []
        self.losses = []

    def __call__(self, score: float, loss: float) -> None:
        """
        Update the callback with the new score (from a finished episode).

        Args:
            score: the score at the end of any episode to log
            loss: the loss from training the network

        Returns:
            None

        """
        # append the score to the list
        self.scores.append(score)
        self.losses.append(loss)
        # plot the score
        plt.subplot(2, 1, 1)
        plt.plot(self.scores)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        # plot the loss
        plt.subplot(2, 1, 2)
        plt.plot(self.losses)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        # adjust the layout
        plt.tight_layout()
        # clear the Jupyter front-end and send the new plot
        display.clear_output(wait=True)
        plt.show()


# explicitly define the outward facing API of this module
__all__ = ['JupyterCallback']
