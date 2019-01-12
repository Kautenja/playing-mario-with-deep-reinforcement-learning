"""The Command Line Interface (CLI) for the project."""
from .args_and_groups import args_and_groups
from .build_argparser import build_argparser


# explicitly define the outward facing API of the module
__all__ = [
    args_and_groups.__name__,
    build_argparser.__name__,
]
