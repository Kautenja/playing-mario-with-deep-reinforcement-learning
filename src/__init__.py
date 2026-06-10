"""Deprecated legacy source tree from the original Keras implementation."""
from __future__ import annotations

import warnings


DEPRECATION_MESSAGE = (
    "The legacy src package is deprecated and kept only for historical "
    "reference. Use the supported mario_rl package, python -m mario_rl.*, "
    "or ./main.sh commands instead."
)


def deprecated_error(command: str | None = None) -> str:
    """Return a deprecation error message for legacy command shims."""
    prefix = f"{command} is deprecated. " if command else ""
    return f"{prefix}{DEPRECATION_MESSAGE}"


warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


__all__ = ["DEPRECATION_MESSAGE", "deprecated_error"]
