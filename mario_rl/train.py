"""Config-driven training entrypoint placeholder for the PyTorch port."""
from __future__ import annotations

import json
from collections.abc import Sequence

from .config import MarioRLConfig, cli, to_dict


def run(config: MarioRLConfig) -> int:
    """Report the resolved config until the Lightning training spec owns work."""
    print(json.dumps({"command": "train", "config": to_dict(config)}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse training config and execute the training command."""
    return cli(
        argv,
        description="Train a Mario DQN experiment from a typed config.",
        runner=run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
