"""Command entrypoint for the modern Mario RL package."""
from . import __version__
from .config import available_configs


def main() -> int:
    """Print concise command help."""
    configs = ", ".join(available_configs())
    print(f"mario_rl {__version__}")
    print("commands: config, train, play, random, verify-macbook")
    print(f"packaged configs: {configs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
