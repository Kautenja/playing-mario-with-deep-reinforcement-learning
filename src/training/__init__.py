"""Modules for training models."""
from .stochastic import stochastic


# explicitly define the outward facing API of this package
__all__ = [
    stochastic.__name__,
]
