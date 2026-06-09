"""Config-driven play and evaluation entrypoint."""
from __future__ import annotations

import json
from collections.abc import Sequence

from .config import MarioRLConfig, cli, to_dict


def run(config: MarioRLConfig) -> int:
    """Report the resolved evaluation config until checkpoint loading lands."""
    print(json.dumps({"command": "play", "config": to_dict(config)}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse play/evaluation config and execute the command."""
    return cli(
        argv,
        description="Evaluate a Mario DQN checkpoint from a typed config.",
        runner=run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
