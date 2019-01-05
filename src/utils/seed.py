"""A method to seed the random number generator."""
import random
import numpy as np
import tensorflow as tf


def seed(seed: int) -> None:
    """
    Seed the random number generators in the project.

    Args:
        seed: the seed for the random number generators

    Returns:
        None

    """
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


# explicitly define the outward facing API of this module
__all__ = [seed.__name__]
