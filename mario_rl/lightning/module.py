"""LightningModule implementation for online replay-driven Mario DQN training."""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningModule

from mario_rl.config import MarioRLConfig, to_dict
from mario_rl.envs import TaskFeatureEncoder
from mario_rl.lightning.data import build_step_dataloader
from mario_rl.models import build_model, compute_dqn_loss, compute_td_targets, make_optimizer
from mario_rl.replay import UniformReplayBuffer, build_replay_buffer
from mario_rl.schedules import EpsilonGreedyActionSelector, LinearEpsilonSchedule


EnvFactory = Callable[[MarioRLConfig], Any]


class DQNLightningModule(LightningModule):
    """Online DQN trainer with explicit replay and target-network ownership."""

    def __init__(
        self,
        config: MarioRLConfig,
        *,
        env_factory: EnvFactory | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._env_factory = env_factory
        self.automatic_optimization = False
        self.save_hyperparameters({"config": to_dict(config)})

        self.q_network = build_model(config)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        for parameter in self.target_q_network.parameters():
            parameter.requires_grad_(False)

        seed = config.trainer.seed if config.trainer.seed is not None else config.env.seed
        self.replay: UniformReplayBuffer = build_replay_buffer(config, seed=seed)
        self.epsilon_schedule = LinearEpsilonSchedule(
            start=config.epsilon.start,
            final=config.epsilon.final,
            decay_frames=config.epsilon.decay_frames,
        )
        self.action_selector = EpsilonGreedyActionSelector(
            num_actions=config.model.num_actions,
            seed=seed,
        )
        self.task_encoder = None
        self._task_features: np.ndarray | None = None
        self._configure_task_features()

        self.env = None
        self._last_state: np.ndarray | None = None
        self.env_frames = 0
        self.episodes = 0
        self.episode_reward = 0.0
        self.last_loss = 0.0
        self.training_updates = 0

    def train_dataloader(self):
        """Use a bounded placeholder stream so Lightning owns fit iteration."""
        return build_step_dataloader(self.config.train.max_steps)

    def configure_optimizers(self):
        """Create the configured optimizer for the online Q-network."""
        return make_optimizer(self.q_network, self.config)

    def training_step(self, _batch, _batch_idx):
        """Collect one transition, optionally optimize from replay, and log metrics."""
        self._ensure_env()
        reward = self._collect_transition()
        loss = self._optimize_from_replay()
        if loss is None:
            loss_tensor = torch.zeros((), device=self.device)
        else:
            loss_tensor = loss.detach()
            self.last_loss = float(loss_tensor.cpu().item())

        epsilon = self.epsilon_schedule.value()
        lr = self._current_learning_rate()
        self.log("train/loss", loss_tensor, on_step=True, prog_bar=False)
        self.log("train/episode_reward", self.episode_reward, on_step=True, prog_bar=False)
        self.log("train/epsilon", epsilon, on_step=True, prog_bar=False)
        self.log("train/env_frames", float(self.env_frames), on_step=True, prog_bar=False)
        if lr is not None:
            self.log("train/learning_rate", lr, on_step=True, prog_bar=False)

        return {
            "loss": loss_tensor,
            "reward": torch.tensor(float(reward), device=self.device),
        }

    def on_train_end(self) -> None:
        """Close the environment at the end of fit."""
        self.close_env()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Add schedule and environment counters to Lightning checkpoints."""
        checkpoint["mario_rl_state"] = {
            "env_frames": int(self.env_frames),
            "episodes": int(self.episodes),
            "episode_reward": float(self.episode_reward),
            "last_loss": float(self.last_loss),
            "training_updates": int(self.training_updates),
            "epsilon_schedule": self.epsilon_schedule.state_dict(),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore explicit schedule and counter state from a checkpoint."""
        state = checkpoint.get("mario_rl_state", {})
        self.env_frames = int(state.get("env_frames", self.env_frames))
        self.episodes = int(state.get("episodes", self.episodes))
        self.episode_reward = float(state.get("episode_reward", self.episode_reward))
        self.last_loss = float(state.get("last_loss", self.last_loss))
        self.training_updates = int(state.get("training_updates", self.training_updates))
        if "epsilon_schedule" in state:
            self.epsilon_schedule.load_state_dict(state["epsilon_schedule"])

    def close_env(self) -> None:
        """Close the active environment if one has been created."""
        if self.env is not None:
            self.env.close()
            self.env = None
            self._last_state = None

    def metrics_summary(self) -> dict[str, float | int]:
        """Return stable final metrics for command-level artifact writing."""
        return {
            "global_step": int(self.global_step),
            "env_frames": int(self.env_frames),
            "episodes": int(self.episodes),
            "episode_reward": float(self.episode_reward),
            "epsilon": float(self.epsilon_schedule.value()),
            "loss": float(self.last_loss),
            "learning_rate": float(self.config.model.learning_rate),
        }

    def _ensure_env(self) -> None:
        if self.env is not None and self._last_state is not None:
            return
        if self._env_factory is None:
            from mario_rl.envs import make_env

            self.env = make_env(config=self.config.env.to_mario_env_config())
        else:
            self.env = self._env_factory(self.config)
        state, _ = self.env.reset(seed=self.config.env.seed)
        self._last_state = self._coerce_state(state)
        self.episode_reward = 0.0

    def _collect_transition(self) -> float:
        assert self.env is not None
        assert self._last_state is not None

        epsilon = self.epsilon_schedule.value(self.env_frames)
        state_tensor = torch.as_tensor(
            self._last_state,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(
                state_tensor,
                self._task_feature_tensor(batch_size=1),
            ).squeeze(0).detach().cpu()
        action = self.action_selector.select(q_values, epsilon=epsilon)

        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = self._coerce_state(next_state)
        self.replay.push(
            self._last_state,
            action,
            float(reward),
            bool(terminated),
            bool(truncated),
            next_state,
            task_features=self._task_features,
            next_task_features=self._task_features,
        )

        frames = int(info.get("frames_skipped", 1)) if isinstance(info, dict) else 1
        self.env_frames += max(frames, 1)
        self.epsilon_schedule.current_step = self.env_frames
        self.episode_reward += float(reward)

        if terminated or truncated:
            self.episodes += 1
            reset_state, _ = self.env.reset(seed=self.config.env.seed)
            self._last_state = self._coerce_state(reset_state)
            self.episode_reward = 0.0
        else:
            self._last_state = next_state
        return float(reward)

    def _optimize_from_replay(self) -> torch.Tensor | None:
        if len(self.replay) < int(self.config.replay.warmup):
            return None

        batch = self.replay.sample(
            int(self.config.replay.batch_size),
            device=self.device,
            as_tensors=True,
        )
        optimizer = self.optimizers()
        q_values = self.q_network(batch.state, batch.task_features)
        with torch.no_grad():
            target_next_q = self.target_q_network(batch.next_state, batch.next_task_features)
            online_next_q = None
            if self.config.model.double_dqn:
                online_next_q = self.q_network(batch.next_state, batch.next_task_features)
            targets = compute_td_targets(
                batch.reward,
                batch.terminated,
                batch.truncated,
                target_next_q,
                discount_factor=self.config.model.discount_factor,
                online_next_q_values=online_next_q,
                double_dqn=self.config.model.double_dqn,
            )

        loss = compute_dqn_loss(q_values, batch.action, targets)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        self.training_updates += 1
        if self.training_updates % int(self.config.model.target_update_frequency) == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        return loss

    def _configure_task_features(self) -> None:
        feature_size = int(getattr(self.q_network, "task_feature_size", 0))
        if feature_size <= 0:
            return
        self.task_encoder = TaskFeatureEncoder()
        if self.task_encoder.feature_size != feature_size:
            raise ValueError(
                "configured task feature size "
                f"{feature_size} does not match encoder size {self.task_encoder.feature_size}"
            )
        self._task_features = self.task_encoder.encode_env_id(self.config.env.id).vector

    def _task_feature_tensor(self, *, batch_size: int) -> torch.Tensor | None:
        if self._task_features is None:
            return None
        tensor = torch.as_tensor(
            self._task_features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        if batch_size != 1:
            tensor = tensor.expand(batch_size, -1)
        return tensor

    def _coerce_state(self, state) -> np.ndarray:
        array = np.asarray(state, dtype=np.dtype(self.config.replay.sample_dtype))
        expected = tuple(self.config.replay.state_shape)
        if array.shape != expected:
            raise ValueError(f"expected observation shape {expected}, got {array.shape}")
        return array

    def _current_learning_rate(self) -> float | None:
        optimizers = self.optimizers(use_pl_optimizer=False)
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        if optimizer is None:
            return None
        return float(optimizer.param_groups[0]["lr"])


__all__ = ["DQNLightningModule", "EnvFactory"]
