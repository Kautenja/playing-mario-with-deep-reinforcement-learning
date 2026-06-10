"""Checkpoint evaluation helpers for Mario DQN experiments."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mario_rl.config import MarioRLConfig, action_space_summary, with_resolved_model_num_actions
from mario_rl.envs import TaskFeatureEncoder
from mario_rl.lightning.artifacts import checkpoint_path, experiment_paths, write_json
from mario_rl.lightning.module import DQNLightningModule
from mario_rl.metrics import MarioMetricsAccumulator
from mario_rl.rewards import RewardTransformer
from mario_rl.schedules import EpsilonGreedyActionSelector


EnvFactory = Callable[[MarioRLConfig], Any]


def evaluate_checkpoint(
    config: MarioRLConfig,
    *,
    checkpoint: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    """Load a Lightning checkpoint and run bounded evaluation episodes."""
    config = with_resolved_model_num_actions(config)
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
    task_features = None
    task_feature_size = int(getattr(module.q_network, "task_feature_size", 0))
    if task_feature_size > 0:
        encoder = TaskFeatureEncoder()
        if encoder.feature_size != task_feature_size:
            raise ValueError(
                "checkpoint task feature size "
                f"{task_feature_size} does not match encoder size {encoder.feature_size}"
            )
        task_features = encoder.encode_env_id(config.env.id).to_tensor().unsqueeze(0)
    episode_metrics = []
    env = env_factory(config)
    reward_transformer = RewardTransformer(config.reward_transform)
    metrics = MarioMetricsAccumulator(default_task_id=config.env.id)
    try:
        for episode in range(int(config.eval.episodes)):
            state, reset_info = env.reset(seed=config.env.seed)
            metrics.start_episode(
                reset_info if isinstance(reset_info, dict) else None,
                fallback_env_id=config.env.id,
            )
            total_reward = 0.0
            total_transformed_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            while steps < int(config.eval.max_steps):
                state_array = np.asarray(state, dtype=np.dtype(config.replay.sample_dtype))
                with torch.no_grad():
                    q_values = module.q_network(
                        torch.as_tensor(state_array).unsqueeze(0),
                        task_features,
                    )
                action = selector.select(
                    q_values.squeeze(0).cpu(),
                    epsilon=0.0,
                    deterministic=config.eval.deterministic,
                )
                state, reward, terminated, truncated, info = env.step(action)
                transformed = reward_transformer.transform(
                    float(reward),
                    info if isinstance(info, dict) else None,
                )
                frames = int(info.get("frames_skipped", 1)) if isinstance(info, dict) else 1
                metrics.observe_step(
                    reward=float(reward),
                    transformed=transformed,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info if isinstance(info, dict) else None,
                    fallback_env_id=config.env.id,
                    frame_count=max(frames, 1),
                )
                total_reward += float(reward)
                total_transformed_reward += transformed.training_reward
                steps += 1
                if terminated or truncated:
                    break
            limit_truncated = (
                steps >= int(config.eval.max_steps)
                and not bool(terminated)
                and not bool(truncated)
            )
            metrics.finish_episode(
                terminated=bool(terminated),
                truncated=bool(truncated or limit_truncated),
            )
            episode_metrics.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "transformed_reward": total_transformed_reward,
                    "steps": steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated or limit_truncated),
                }
            )
    finally:
        env.close()

    metrics_payload = metrics.to_payload(include_active=False)
    global_metrics = metrics_payload["global"]
    payload = {
        **action_space_summary(config),
        "checkpoint": str(ckpt_path),
        **metrics_payload,
        "legacy_episodes": episode_metrics,
        "episode_count": len(episode_metrics),
        "total_reward": float(global_metrics["episode_return_total"]),
        "total_transformed_reward": float(global_metrics["transformed_return_total"]),
        "total_steps": int(global_metrics["step_count"]),
    }
    write_json(paths.eval_metrics, payload)
    payload["metrics_path"] = str(paths.eval_metrics)
    return payload


__all__ = ["evaluate_checkpoint"]
