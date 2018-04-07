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

    def push(self, s, a, r, d, s2) -> None:
        """
        Push a new experience onto the queue.

        Args:
            s: the current state
            a: the action to get from state to next state
            r: the reward as a result of the action
            d: a flag indicating if the episode (game) has ended
            s2: the next state as a result of the action from state

        Returns:
            None

        """
        # ensure types are the smallest possible before storing in the queue
        s = s.astype('uint8')
        s2 = s2.astype('uint8')
        a = int(a)
        r = int(r)
        # push the variables onto the queue
        self.queue.append((s, a, r, d, s2))

    def deque(self) -> tuple:
        """Pop an item off the queue and return it."""
        return self.queue.popleft()

    def sample(self, size: int=32):
        """
        Return a random sample of items from the queue.

        Args:
            size: the number of items to sample and return

        Returns:
            A random sample from the queue sampled uniformly

        """
        # generate an index of items to extract
        idx_batch = set(np.random.randint(0, len(self), size))
        # extract the batch from the queue
        return [val for i, val in enumerate(self.queue) if i in idx_batch]





# explicitly define the outward facing API of this module
__all__ = ['ReplayQueue']
