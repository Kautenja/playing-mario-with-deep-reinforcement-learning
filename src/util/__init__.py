"""Legacy utilities for the project."""

__all__ = ["BaseCallback", "JupyterCallback"]


def __getattr__(name):
    """Lazily import utilities and optional plotting dependencies."""
    if name == "BaseCallback":
        from .base_callback import BaseCallback

        return BaseCallback
    if name == "JupyterCallback":
        from .jupyter_callback import JupyterCallback

        return JupyterCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
