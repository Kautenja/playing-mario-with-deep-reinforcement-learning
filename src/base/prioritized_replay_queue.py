"""A priority queue for storing previous experiences to sample from."""
import itertools
from heapq import heappush, heappushpop
import numpy as np


class PrioritizedReplayQueue(object):
    """A prioritized replay queue for replaying previous experiences."""

    def __init__(self, size: int) -> None:
        """
        Initialize a new prioritized replay buffer with a given size.

        Args:
            size: the max number of experiences to store in the queue

        Returns:
            None

        """
        # type check the size parameter
        if not isinstance(size, int):
            raise TypeError('`size` must be of type int')
        # ensure the size is within a legal range of values
        if size <= 0:
            raise ValueError('`size` must be > 0')
        self.size = size
        # initialize the priority queue as a heap
        self.heap = []
        # create a counter for resolving issues of equal priority. multi-
        # dimensional numpy arrays can't be directly compared, thus rendering
        # a value error if we try to compare them. Thus adding a unique count
        # as the secondary comparison (after priority) prevents the comparison
        # of numpy arrays altogether
        self.counter = itertools.count()

    def __repr__(self) -> str:
        """Return an executable string representation of priority queue."""
        return '{}(size={})'.format(self.__class__.__name__, self.size)

    @property
    def top(self)-> int:
        """Return the index of the top element in the queue."""
        return len(self.heap)

    def push(self,
        s: np.ndarray,
        a: int,
        r: int,
        d: bool,
        s2: np.ndarray,
        priority: float
    ) -> None:
        """
        Push a new experience onto the queue.

        Args:
            s: the current state
            a: the action to get from current state `s` to next state `s2`
            r: the reward resulting from taking action `a` in state `s`
            d: the flag denoting whether the episode ended after action `a`
            s2: the next state from taking action `a` in state `s`
            priority: the priority of the item to push to the queue

        Returns:
            None

        """
        # get the unique count for this item
        count = next(self.counter)
        # if the heap has arrived at capacity, use push pop to add new items
        if len(self.heap) == self.size:
            heappushpop(self.heap, (priority, count, (s, a, r, d, s2)))
        # otherwise heap push the item onto the queue
        else:
            heappush(self.heap, (priority, count, (s, a, r, d, s2)))

    def sample(self, size: int=32) -> bool:
        """
        Return a random sample of items from the queue.

        Args:
            size: the number of items to sample and return

        Returns:
            A random sample from the queue sampled uniformly

        """
        # extract a sample from the heap (priorities are in increasing order)
        # i.e. the lowest priority value is the first item in the sample.
        # ignore the first two values in each heap item (priority & count)
        sample_batch = [experience for (_, _, experience) in self.heap[-size:]]
        # initialize lists for each component of the batch
        s = [None] * len(sample_batch)
        a = [None] * len(sample_batch)
        r = [None] * len(sample_batch)
        d = [None] * len(sample_batch)
        s2 = [None] * len(sample_batch)
        # iterate over the indexes and copy references to the arrays
        for batch, sample in enumerate(sample_batch):
            _s, _a, _r, _d, _s2 = sample
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
__all__ = [PrioritizedReplayQueue.__name__]
