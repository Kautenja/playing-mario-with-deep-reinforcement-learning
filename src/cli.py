"""Deprecated legacy CLI shim."""
from __future__ import annotations

from collections.abc import Sequence

from . import deprecated_error


def main(argv: Sequence[str] | None = None) -> None:
    """Reject the legacy CLI with a replacement path."""
    del argv
    raise SystemExit(deprecated_error("python -m src.cli"))


if __name__ == "__main__":
    main()


# explicitly define the outward facing API of this module
__all__ = [main.__name__]
