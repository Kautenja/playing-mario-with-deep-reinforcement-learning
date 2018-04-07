"""A queue for storing previous experiences to sample from."""
import numpy as np


class ReplayQueue(object):
    """A queue for storing previous experiences to sample from."""

    def __init__(self,
        size: int=250000,
        image_size: tuple=(84, 84),
        history_length: int=4
    ) -> None:
        """
        Initialize a new replay buffer with a given size.

        Args:
            size: the maximum number of experiences to store
            image_size: the size of the images to store
            history_length: the length of frame history for an agent

        Returns:
            None

        """
        # assign size to self
        self._size = size
        # setup variables for the index and top
        self.index = 0
        self.top = 0
        # setup the queues
        self.s = np.zeros((size, *image_size, history_length), dtype=np.uint8)
        self.a = np.zeros(size, dtype=np.uint8)
        self.r = np.zeros(size, dtype=np.int8)
        self.d = np.zeros(size, dtype=np.bool)
        self.s2 = np.zeros((size, *image_size, history_length), dtype=np.uint8)

    def __repr__(self) -> str:
        """Return an executable string representation of self."""
        return '{}(size={})'.format(self.__class__.__name__, self.size)

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return self.top

    @property
    def size(self) -> int:
        """Return the size of the queue."""
        return self._size

    @property
    def num_bytes(self) -> int:
        """Return the number of byte this object consumes."""
        from sys import getsizeof
        s = getsizeof(self.s)
        a = getsizeof(self.a)
        r = getsizeof(self.r)
        d = getsizeof(self.d)
        s2 = getsizeof(self.s2)

        return s + a + r + d + s2

    def push(self,
        s: np.ndarray,
        a: np.uint8,
        r: np.int8,
        d: np.bool,
        s2: np.ndarray
    ) -> None:
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
        # push the variables onto the queue
        self.s[self.index] = s
        self.a[self.index] = a
        self.r[self.index] = r
        self.d[self.index] = d
        self.s2[self.index] = s2
        # increment the index
        self.index = (self.index + 1) % self.size
        # increment the top pointer
        if self.top < self.size:
            self.top += 1

    def current(self) -> tuple:
        """Pop an item off the queue and return it."""
        s = self.s[self.index]
        a = self.a[self.index]
        r = self.r[self.index]
        d = self.d[self.index]
        s2 = self.s2[self.index]

        return s, a, r, d, s2

    def sample(self, size: int=32) -> tuple:
        """
        Return a random sample of items from the queue.

        Args:
            size: the number of items to sample and return

        Returns:
            A random sample from the queue sampled uniformly

        """
        # generate and index of items to sample
        index = np.random.randint(0, self.top, size)
        # extract the items for this batch
        s = self.s[index]
        a = self.a[index]
        r = self.r[index]
        d = self.d[index]
        s2 = self.s2[index]

        return s, a, r, d, s2


# explicitly define the outward facing API of this module
__all__ = ['ReplayQueue']
