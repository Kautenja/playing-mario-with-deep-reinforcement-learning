"""Config-driven random rollout entrypoint."""
from __future__ import annotations

import json
from collections.abc import Sequence

from .config import MarioRLConfig, action_space_summary, cli


def run(config: MarioRLConfig) -> int:
    """Run a bounded random rollout for smoke checks."""
    from mario_rl.envs import make_env

    env = make_env(config=config.env.to_mario_env_config())
    action_summary = action_space_summary(config, env=env)
    total_reward = 0.0
    steps = 0
    try:
        env.reset(seed=config.env.seed)
        while steps < config.env.max_smoke_steps:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    print(
        json.dumps(
            {
                "command": "random",
                **action_summary,
                "steps": steps,
                "reward": total_reward,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse random-rollout config and execute the command."""
    return cli(
        argv,
        description="Run a bounded random Mario rollout from a typed config.",
        runner=run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
