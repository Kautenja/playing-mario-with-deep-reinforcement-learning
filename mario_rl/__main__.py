"""Command entrypoint for the modern Mario RL package."""
from . import __version__


def main() -> None:
    """Print the package version until the modern CLI is introduced."""
    print(f"mario_rl {__version__}")


if __name__ == "__main__":
    main()
