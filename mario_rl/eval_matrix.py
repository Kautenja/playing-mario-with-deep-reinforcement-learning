"""Config-driven evaluation matrix entrypoint."""
from __future__ import annotations

import json
from collections.abc import Sequence

from .config import MarioRLConfig, cli


def run(config: MarioRLConfig, *, env_factory=None, policy_factory=None) -> int:
    """Evaluate a checkpoint or injected policy across the configured matrix."""
    from mario_rl.evaluation_matrix import run_evaluation_matrix

    payload = run_evaluation_matrix(
        config,
        env_factory=env_factory,
        policy_factory=policy_factory,
    )
    print(
        json.dumps(
            {
                "command": "eval-matrix",
                "summary_path": payload["summary_path"],
                "table_path": payload["table_path"],
                "task_count": payload["matrix"]["task_count"],
                "row_count": payload["row_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse matrix evaluation config and execute the command."""
    return cli(
        argv,
        description="Evaluate a Mario checkpoint across a task matrix.",
        runner=run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
