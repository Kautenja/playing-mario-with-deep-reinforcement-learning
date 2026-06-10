"""Deprecated legacy training shim."""
from __future__ import annotations

from . import deprecated_error


def train(*args, **kwargs) -> None:
    """Reject legacy Keras training and point callers to the PyTorch path."""
    del args, kwargs
    raise RuntimeError(deprecated_error("./main.sh train"))


# explicitly define the outward facing API of this module
__all__ = [train.__name__]
