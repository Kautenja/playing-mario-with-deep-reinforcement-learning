"""A replay buffer for agents."""
from collections import deque
import numpy as np


class ReplayQueue(object):
    """A replay queue for replaying previous experiences."""

    def __init__(self, size: int=20000) -> None:
        """
        Initialize a new replay buffer with a given size.

        Args:
            size: the size of the replay buffer
                  (the number of previous experiences to store)

        Returns:
            None

        """
        # verify size
        if not isinstance(size, int):
            raise TypeError('size must be of type int')
        if size < 1:
            raise ValueError('size must be at least 1')
        # initialize the queue data-structure
        self.queue = deque(maxlen=size)

    def __repr__(self) -> str:
        """Return an executable string representation of self."""
        return '{}(size={})'.format(self.__class__.__name__, self.queue.maxlen)

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self.queue)

    @property
    def size(self) -> int:
        """Return the size of the queue."""
        return self.queue.maxlen

    def push(self, *args) -> None:
        """
        Push a new experience onto the queue.

        Args:
            *args: the experience s, a, r, d, s'

        Returns:
            None

        """
        self.queue.append(args)

    def dequeu(self) -> tuple:
        """Pop an item off the queue and return it."""
        return self.queue.popleft()

    def sample(self, size: int=64, replace: bool=True):
        """
        Return a random sample of items from the queue.

        Args:
            size: the number of items to sample and return

        Returns:
            TODO

        """
        # generate an index of items to extract
        idx_batch = set(np.random.randint(0, len(self), size))
        # extract the batch from the queue
        return [val for i, val in enumerate(self.queue) if i in idx_batch]


# explicitly define the outward facing API of this module
__all__ = ['ReplayQueue']
