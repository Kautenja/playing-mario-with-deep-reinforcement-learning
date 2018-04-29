"""A rich reward tracking callback for Jupyter notebooks."""
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython import display


class JupyterCallback(object):
    """A rich reward tracking callback for Jupyter notebooks."""

    def __init__(self, width: float=14, height_per_plot: float=2.5) -> None:
        """
        Create a new Jupyter Callback method.

        Args:
            width: the width of the plot to render
            height_per_plot: the height of each individual plot

        Returns:
            None

        """
        # setup caches for metrics
        self.scores = []
        self.losses = []
        # create a list of tuples for plotting data
        self.metrics = [
            (self.scores, 'Reward'),
            (self.losses, 'Loss')
        ]
        # set the figsize of this callback
        self.figsize = width, height_per_plot * len(self.metrics)

    def __call__(self, agent, score: float, loss: float) -> None:
        """
        Update the callback with the new score (from a finished episode).

        Args:
            agent: the agent producing the score and loss
            score: the score at the end of any episode to log
            loss: the loss from training the network

        Returns:
            None

        """
        # append the score to the list
        self.scores.append(score)
        self.losses.append(loss)
        # create a figure
        plt.figure(figsize=self.figsize)
        for index, (metric, ylabel) in enumerate(self.metrics):
            plt.subplot(len(self.metrics), 1, index + 1)
            plt.plot(metric)
            plt.xlabel('Episode')
            plt.ylabel(ylabel)
            # force integer axis tick labels
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # adjust the layout
        plt.tight_layout()
        # clear the Jupyter front-end and send the new plot
        display.clear_output(wait=True)
        plt.show()


# explicitly define the outward facing API of this module
__all__ = [JupyterCallback.__name__]
