"""A rich reward tracking callback for Jupyter notebooks."""
from matplotlib import pyplot as plt
from IPython import display


class JupyterCallback(object):
    """A rich reward tracking callback for Jupyter notebooks."""

    def __init__(self,
                 xlabel: str='Episode',
                 ylabel: str='Score'
        ) -> None:
        """
        Create a new Jupyter Callback method.

        Args:
            xlabel: the label to show on the x axis
            ylabel: the label to show on the y axis

        Returns:
            None

        """
        # verify xlabel
        if not isinstance(xlabel, str):
            raise TypeError('xlabel must be of type: str')
        # verify ylabel
        if not isinstance(ylabel, str):
            raise TypeError('ylabel must be of type: str')
        # assign arguments to self
        self.xlabel = xlabel
        self.ylabel = ylabel
        # setup instance members
        self.scores = []

    def __call__(self, score: float) -> None:
        """
        Update the callback with the new score (from a finished episode).

        Args:
            score: the score at the end of any episode to log

        Returns:
            None

        """
        # append the score to the list
        self.scores.append(score)
        # plot the score
        plt.plot(self.scores)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        # clear the Jupyter front-end and send the new plot
        display.clear_output(wait=True)
        plt.show()


# explicitly define the outward facing API of this module
__all__ = ['JupyterCallback']
