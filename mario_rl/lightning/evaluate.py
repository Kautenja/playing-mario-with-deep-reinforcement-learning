"""Checkpoint evaluation helpers for Mario DQN experiments."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mario_rl.config import MarioRLConfig
from mario_rl.lightning.artifacts import checkpoint_path, experiment_paths, write_json
from mario_rl.lightning.module import DQNLightningModule
from mario_rl.schedules import EpsilonGreedyActionSelector


EnvFactory = Callable[[MarioRLConfig], Any]


def evaluate_checkpoint(
    config: MarioRLConfig,
    *,
    checkpoint: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    """Load a Lightning checkpoint and run bounded evaluation episodes."""
    paths = experiment_paths(config)
    ckpt_path = Path(checkpoint).expanduser() if checkpoint else checkpoint_path(config, paths)
    module = DQNLightningModule.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        env_factory=env_factory,
        map_location="cpu",
    )
    module.eval()

    if env_factory is None:
        from mario_rl.envs import make_env

        def env_factory(config: MarioRLConfig):
            env_config = config.env
            if env_config.video_enabled and env_config.video_dir is None:
                env_config = replace(
                    env_config,
                    render_mode=env_config.render_mode or "rgb_array",
                    video_dir=str(paths.videos),
                )
            return make_env(config=env_config.to_mario_env_config())

    selector = EpsilonGreedyActionSelector(
        num_actions=config.model.num_actions,
        seed=config.trainer.seed if config.trainer.seed is not None else config.env.seed,
    )
    episode_metrics = []
    env = env_factory(config)
    try:
        for episode in range(int(config.eval.episodes)):
            state, _ = env.reset(seed=config.env.seed)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            while steps < int(config.eval.max_steps):
                state_array = np.asarray(state, dtype=np.dtype(config.replay.sample_dtype))
                with torch.no_grad():
                    q_values = module.q_network(torch.as_tensor(state_array).unsqueeze(0))
                action = selector.select(
                    q_values.squeeze(0).cpu(),
                    epsilon=0.0,
                    deterministic=config.eval.deterministic,
                )
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            episode_metrics.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    finally:
        env.close()

    payload = {
        "checkpoint": str(ckpt_path),
        "episodes": episode_metrics,
        "episode_count": len(episode_metrics),
        "total_reward": float(sum(item["reward"] for item in episode_metrics)),
        "total_steps": int(sum(item["steps"] for item in episode_metrics)),
    }
    write_json(paths.eval_metrics, payload)
    payload["metrics_path"] = str(paths.eval_metrics)
    return payload


__all__ = ["evaluate_checkpoint"]
