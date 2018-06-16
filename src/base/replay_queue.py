"""A queue for storing previous experiences to sample from."""
import numpy as np


class ReplayQueue(object):
    """A replay queue for replaying previous experiences."""

    def __init__(self, size: int) -> None:
        """
        Initialize a new replay buffer with a given size.

        Args:
            size: the size of the replay buffer
                  (the number of previous experiences to store)

        Returns:
            None

        """
        # initialize the queue data-structure as a list of nil values
        self.queue = [None] * size
        # setup variables for the index and top
        self.index = 0
        self.top = 0

    def __repr__(self) -> str:
        """Return an executable string representation of self."""
        return '{}(size={})'.format(self.__class__.__name__, self.size)

    @property
    def size(self) -> int:
        """Return the size of the queue."""
        return len(self.queue)

    def push(self,
        s: np.ndarray,
        a: int,
        r: int,
        d: bool,
        s2: np.ndarray,
    ) -> None:
        """
        Push a new experience onto the queue.

        Args:
            s: the current state
            a: the action to get from current state `s` to next state `s2`
            r: the reward resulting from taking action `a` in state `s`
            d: the flag denoting whether the episode ended after action `a`
            s2: the next state from taking action `a` in state `s`

        Returns:
            None

        """
        # push the variables onto the queue
        self.queue[self.index] = s, a, r, d, s2
        # increment the index
        self.index = (self.index + 1) % self.size
        # increment the top pointer
        if self.top < self.size:
            self.top += 1

    def sample(self, size: int=32) -> tuple:
        """
        Return a random sample of items from the queue.

        Args:
            size: the number of items to sample and return

        Returns:
            A random sample from the queue sampled uniformly

        """
        # initialize lists for each component of the batch
        s = [None] * size
        a = [None] * size
        r = [None] * size
        d = [None] * size
        s2 = [None] * size
        # iterate over the indexes and copy references to the arrays
        for batch, sample in enumerate(np.random.randint(0, self.top, size)):
            _s, _a, _r, _d, _s2 = self.queue[sample]
            s[batch] = np.array(_s, copy=False)
            a[batch] = _a
            r[batch] = _r
            d[batch] = _d
            s2[batch] = np.array(_s2, copy=False)
        # convert the lists to arrays for returning for training
        return (
            np.array(s),
            np.array(a, dtype=np.uint8),
            np.array(r, dtype=np.int8),
            np.array(d, dtype=np.bool),
            np.array(s2),
        )


# explicitly define the outward facing API of this module
__all__ = [ReplayQueue.__name__]
