"""Deprecated legacy play/random shims."""
from __future__ import annotations

from . import deprecated_error


def plot_results(*args, **kwargs) -> None:
    """Reject the legacy Keras result plotting helper."""
    del args, kwargs
    raise RuntimeError(deprecated_error("src.play.plot_results"))


def play(*args, **kwargs) -> None:
    """Reject legacy Keras checkpoint playback and point callers to PyTorch."""
    del args, kwargs
    raise RuntimeError(deprecated_error("./main.sh play"))


def play_random(*args, **kwargs) -> None:
    """Reject legacy random rollout and point callers to the modern command."""
    del args, kwargs
    raise RuntimeError(deprecated_error("./main.sh random"))


# explicitly define the outward facing API of this module
__all__ = [plot_results.__name__, play.__name__, play_random.__name__]
