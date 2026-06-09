"""Modern Mario reinforcement learning package."""
from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("playing-mario-with-deep-reinforcement-learning")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = ["__version__"]
