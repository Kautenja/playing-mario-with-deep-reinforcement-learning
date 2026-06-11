"""Shared deterministic fake env helpers for fast Mario RL tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from mario_rl.config import MarioRLConfig, load, resolve_model_num_actions


class FakeMarioEnv(gym.Env):
    """Deterministic channel-first image env with Gymnasium step semantics."""

    action_space = gym.spaces.Discrete(12)

    def __init__(
        self,
        episode_length: int = 4,
        *,
        env_id: str = "FakeMario-v0",
        game_family: str = "unknown",
        observation_shape: tuple[int, int, int] = (4, 84, 84),
        num_actions: int = 12,
    ) -> None:
        self.episode_length = int(episode_length)
        self.env_id = str(env_id)
        self.game_family = str(game_family)
        self.action_space = gym.spaces.Discrete(int(num_actions))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=tuple(int(dimension) for dimension in observation_shape),
            dtype=np.uint8,
        )
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.step_count = 0
        return self._obs(0), {
            "env_id": self.env_id,
            "game_family": self.game_family,
            "seed": seed,
            "task_id": self.env_id,
            "options": options,
        }

    def step(self, action):
        self.step_count += 1
        reward = float((int(action) % 3) - 1)
        terminated = self.step_count >= self.episode_length
        truncated = False
        clipped_reward = max(-15.0, min(15.0, reward))
        progress = float(self.step_count)
        return (
            self._obs(self.step_count),
            reward,
            terminated,
            truncated,
            {
                "frames_skipped": 1,
                "fake_step": self.step_count,
                "clear": bool(terminated),
                "death": False,
                "env_id": self.env_id,
                "game_family": self.game_family,
                "progress": progress,
                "progress_max": max(progress, 1.0),
                "task_id": self.env_id,
                "reward_components": {
                    "progress": reward,
                    "death": 0.0,
                },
                "reward_total_unclipped": reward,
                "reward_total_clipped": clipped_reward,
            },
        )

    def dump_state(self):
        """Return an opaque fake snapshot of the current env state."""
        return {
            "env_id": self.env_id,
            "game_family": self.game_family,
            "step_count": self.step_count,
        }

    def load_state(self, snapshot):
        """Restore a fake snapshot captured by :meth:`dump_state`."""
        if not isinstance(snapshot, dict) or snapshot.get("env_id") != self.env_id:
            raise ValueError("incompatible fake snapshot")
        self.step_count = int(snapshot["step_count"])

    def _obs(self, value: int):
        return np.full(self.observation_space.shape, value % 256, dtype=np.uint8)


def fake_env_factory(_config: MarioRLConfig) -> FakeMarioEnv:
    """Create the fake env used by training and evaluation tests."""
    return FakeMarioEnv(
        env_id=_config.env.id,
        game_family=_fake_game_family(_config.env.id),
        observation_shape=tuple(_config.replay.state_shape),
        num_actions=resolve_model_num_actions(_config),
    )


def _fake_game_family(env_id: str) -> str:
    if "Bros3" in env_id:
        return "smb3"
    if "Bros2" in env_id:
        return "smb2_usa"
    if "LostLevels" in env_id:
        return "lost_levels"
    if "SuperMarioBros" in env_id:
        return "smb1"
    return "unknown"


def tiny_training_config(save_dir: str | Path) -> MarioRLConfig:
    """Return a fast CPU config that exercises replay and checkpoints."""
    config = load("smb_dqn_fast_dev")
    return replace(
        config,
        experiment_name="fake_lightning",
        save_dir=str(save_dir),
        trainer=replace(config.trainer, accelerator="cpu", devices=1, seed=123),
        env=replace(
            config.env,
            id="FakeMario-v0",
            render_mode=None,
            frame_skip=1,
            max_smoke_steps=8,
        ),
        replay=replace(
            config.replay,
            capacity=32,
            batch_size=2,
            warmup=2,
            state_shape=(4, 84, 84),
        ),
        model=replace(
            config.model,
            hidden_size=64,
            target_update_frequency=2,
        ),
        epsilon=replace(config.epsilon, decay_frames=8),
        train=replace(
            config.train,
            max_frames=8,
            max_steps=5,
            log_interval=1,
            accelerator="cpu",
            devices=1,
            checkpoint_name="fake.ckpt",
        ),
        eval=replace(config.eval, episodes=1, max_steps=4, checkpoint=None),
    )


def tiny_ppo_config(save_dir: str | Path) -> MarioRLConfig:
    """Return a fast CPU config that exercises recurrent PPO without ROMs."""
    config = tiny_training_config(save_dir)
    return replace(
        config,
        experiment_name="fake_ppo_lightning",
        model=replace(
            config.model,
            architecture="recurrent_actor_critic",
            hidden_size=64,
            recurrent_hidden_size=32,
            task_conditioning=True,
            task_feature_size=0,
        ),
        ppo=replace(
            config.ppo,
            rollout_steps=4,
            minibatch_size=2,
            epochs=1,
            max_grad_norm=0.5,
        ),
        train=replace(
            config.train,
            algorithm="ppo",
            max_frames=8,
            max_steps=2,
            checkpoint_name="fake-ppo.ckpt",
        ),
    )
