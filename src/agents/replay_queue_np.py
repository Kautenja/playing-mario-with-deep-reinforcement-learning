"""A queue for storing previous experiences to sample from."""
import numpy as np


class ReplayQueue(object):
    """A queue for storing previous experiences to sample from."""

    def __init__(self,
        size: int=250000,
        image_size: tuple=(84, 84),
        agent_history_length: int=4
    ) -> None:
        """
        Initialize a new replay buffer with a given size.

        Args:
            size: the maximum number of experiences to store
            image_size: the size of the images to store
            agent_history_length

        Returns:
            None

        """
        # verify size
        if not isinstance(size, int):
            raise TypeError('size must be of type int')
        if size < 1:
            raise ValueError('size must be at least 1')
        # assign size to self
        self._size = size
        # the size of frames
        frame_size = (*image_size, agent_history_length)
        # setup the queues
        self.s = np.zeros((size, *frame_size)).astype('uint8')
        self.a = np.zeros(size).astype('uint8')
        self.r = np.zeros(size).astype('int8')
        self.d = np.zeros(size).astype(bool)
        self.s2 = np.zeros((size, *frame_size)).astype('uint8')
        # setup variables for the index and top
        self.index = 0
        self.top = 0

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
        self.s[self.index] = s.astype('uint8')
        self.a[self.index] = int(a)
        self.r[self.index] = int(r)
        self.d[self.index] = d
        self.s2[self.index] = s2.astype('uint8')
        # increment the index
        if self.index == self.size - 1:
            self.index = 0
        else:
            self.index += 1
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

    def sample(self, size: int=32):
        """
        Return a random sample of items from the queue.

        Args:
            size: the number of items to sample and return

        Returns:
            A random sample from the queue sampled uniformly

        """
        # generate and index of items to sample
        index = np.random.randint(0, len(self), size)
        # extract the items for this batch
        s = self.s[index]
        a = self.a[index]
        r = self.r[index]
        d = self.d[index]
        s2 = self.s2[index]

        return s, a, r, d, s2


# explicitly define the outward facing API of this module
__all__ = ['ReplayQueue']
