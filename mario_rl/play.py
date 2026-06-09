"""Config-driven play and evaluation entrypoint."""
from __future__ import annotations

import json
from collections.abc import Sequence

from .config import MarioRLConfig, cli


def run(config: MarioRLConfig, *, env_factory=None) -> int:
    """Evaluate a Lightning checkpoint and write metrics artifacts."""
    from mario_rl.lightning import evaluate_checkpoint

    payload = evaluate_checkpoint(config, env_factory=env_factory)
    print(json.dumps({"command": "play", **payload}, sort_keys=True))
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
