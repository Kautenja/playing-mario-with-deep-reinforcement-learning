"""Config-driven random rollout entrypoint."""
from __future__ import annotations

if __name__ == "random" and not __package__:
    import importlib.util as _importlib_util
    import os as _os
    import sys as _sys
    import sysconfig as _sysconfig

    _random_path = _os.path.join(_sysconfig.get_path("stdlib"), "random.py")
    _spec = _importlib_util.spec_from_file_location("random", _random_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - interpreter fault.
        raise ImportError(f"could not load stdlib random from {_random_path}")
    _module = _importlib_util.module_from_spec(_spec)
    _sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    import json
    from collections.abc import Sequence

    from .config import MarioRLConfig, action_space_summary, cli

    def run(config: MarioRLConfig, *, env_factory=None) -> int:
        """Run a bounded random rollout for smoke checks."""
        from mario_rl.envs import make_env

        env = (
            env_factory(config)
            if env_factory is not None
            else make_env(config=config.env.to_mario_env_config())
        )
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
