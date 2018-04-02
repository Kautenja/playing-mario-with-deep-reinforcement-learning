"""A rich reward tracking callback for Jupyter notebooks."""
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython import display


class JupyterCallback(object):
    """A rich reward tracking callback for Jupyter notebooks."""

    def __init__(self, max_losses: int=10) -> None:
        """Create a new Jupyter Callback method."""
        self.max_losses = max_losses
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
        # MARK: score
        plt.figure(figsize=(14,5))
        # plot the score
        plt.subplot(2, 1, 1)
        plt.plot(self.scores)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        # force integer axis tick labels
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # MARK: loss
        # keep the losses bounded to a certain size
        losses = self.losses
        if len(losses) > self.max_losses:
            losses = losses[-self.max_losses:]
        # plot the loss
        plt.subplot(2, 1, 2)
        plt.plot(losses)
        plt.xlabel('Last {} Episode(s)'.format(self.max_losses))
        plt.ylabel('Loss')
        # force integer axis tick labels
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # adjust the layout
        plt.tight_layout()
        # clear the Jupyter front-end and send the new plot
        display.clear_output(wait=True)
        plt.show()


# explicitly define the outward facing API of this module
__all__ = ['JupyterCallback']
