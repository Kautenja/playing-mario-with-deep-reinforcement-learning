"""LightningModule implementation for recurrent PPO Mario training."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningModule
from torch.distributions import Categorical

from mario_rl.actor_critic import RolloutStorage
from mario_rl.config import MarioRLConfig, to_dict, with_resolved_model_num_actions
from mario_rl.envs import TaskFeatureEncoder, TaskSuite
from mario_rl.lightning.data import build_step_dataloader
from mario_rl.metrics import MarioMetricsAccumulator
from mario_rl.models import (
    RecurrentActorCritic,
    build_model,
    compute_ppo_loss,
    make_optimizer,
)
from mario_rl.rewards import RewardTransformer


EnvFactory = Callable[[MarioRLConfig], Any]


class PPOLightningModule(LightningModule):
    """On-policy recurrent actor-critic trainer using clipped PPO updates."""

    def __init__(
        self,
        config: MarioRLConfig,
        *,
        env_factory: EnvFactory | None = None,
    ) -> None:
        super().__init__()
        self.config = with_resolved_model_num_actions(config)
        self._env_factory = env_factory
        self.automatic_optimization = False
        self.save_hyperparameters({"config": to_dict(self.config)})

        policy = build_model(self.config)
        if not isinstance(policy, RecurrentActorCritic):
            raise TypeError(
                "PPOLightningModule requires model.architecture='recurrent_actor_critic'"
            )
        self.policy = policy

        self.reward_transformer = RewardTransformer(self.config.reward_transform)
        self.metrics = MarioMetricsAccumulator(default_task_id=self.config.env.id)
        self.task_suite = (
            TaskSuite(self.config.task_suite)
            if bool(getattr(self.config.task_suite, "enabled", False))
            else None
        )
        self.task_encoder = None
        self._task_features: np.ndarray | None = None
        self._configure_task_features()

        self.env = None
        self._active_task_env_id: str | None = None
        self._last_state: np.ndarray | None = None
        self._hidden_state: torch.Tensor | None = None
        self.env_frames = 0
        self.episodes = 0
        self.episode_reward = 0.0
        self.episode_env_reward = 0.0
        self.episode_raw_reward = 0.0
        self.episode_unclipped_reward = 0.0
        self.episode_clipped_reward = 0.0
        self.last_loss = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.last_approximate_kl = 0.0
        self.last_clip_fraction = 0.0
        self.training_updates = 0

    def train_dataloader(self):
        """Use one placeholder item per PPO rollout/update cycle."""
        return build_step_dataloader(self.config.train.max_steps)

    def configure_optimizers(self):
        """Create the configured optimizer for the actor-critic policy."""
        return make_optimizer(self.policy, self.config)

    def training_step(self, _batch, _batch_idx):
        """Collect one rollout and optimize the actor-critic policy with PPO."""
        self._ensure_env()
        rollout = self._collect_rollout()
        next_value = self._next_value()
        rollout.compute_returns_and_advantages(
            next_value,
            discount_factor=self.config.model.discount_factor,
            gae_lambda=self.config.ppo.gae_lambda,
        )
        losses = self._optimize_rollout(rollout)
        self.last_loss = float(losses["total"].detach().cpu().item())
        self.last_policy_loss = float(losses["policy"].detach().cpu().item())
        self.last_value_loss = float(losses["value"].detach().cpu().item())
        self.last_entropy = float(losses["entropy"].detach().cpu().item())
        self.last_approximate_kl = float(losses["approximate_kl"].detach().cpu().item())
        self.last_clip_fraction = float(losses["clip_fraction"].detach().cpu().item())

        self.log("train/loss", losses["total"], on_step=True, prog_bar=False)
        self.log("train/ppo_policy_loss", losses["policy"], on_step=True, prog_bar=False)
        self.log("train/ppo_value_loss", losses["value"], on_step=True, prog_bar=False)
        self.log("train/ppo_entropy", losses["entropy"], on_step=True, prog_bar=False)
        self.log(
            "train/ppo_approximate_kl",
            losses["approximate_kl"],
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/ppo_clip_fraction",
            losses["clip_fraction"],
            on_step=True,
            prog_bar=False,
        )
        self.log("train/episode_reward", self.episode_reward, on_step=True, prog_bar=False)
        self.log(
            "train/episode_env_reward",
            self.episode_env_reward,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/episode_raw_reward",
            self.episode_raw_reward,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/episode_unclipped_reward",
            self.episode_unclipped_reward,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/episode_clipped_reward",
            self.episode_clipped_reward,
            on_step=True,
            prog_bar=False,
        )
        self.log("train/env_frames", float(self.env_frames), on_step=True, prog_bar=False)
        mario_summary = self.metrics.global_summary(include_active=True)
        self.log(
            "train/clear_rate",
            float(mario_summary.clear_rate or 0.0),
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/death_rate",
            float(mario_summary.death_rate or 0.0),
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/truncation_count",
            float(mario_summary.truncation_count),
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/max_progress",
            float(mario_summary.max_progress or 0.0),
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/final_progress_mean",
            float(mario_summary.final_progress_mean or 0.0),
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/learning_rate",
            self.config.model.learning_rate,
            on_step=True,
            prog_bar=False,
        )
        return {
            "loss": losses["total"],
            "reward": torch.as_tensor(
                float(rollout.rewards.sum()),
                device=self.device,
                dtype=torch.float32,
            ),
        }

    def on_train_end(self) -> None:
        """Close the environment at the end of fit."""
        self.close_env()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Add rollout and environment counters to Lightning checkpoints."""
        checkpoint["mario_rl_state"] = {
            "env_frames": int(self.env_frames),
            "episodes": int(self.episodes),
            "episode_reward": float(self.episode_reward),
            "episode_env_reward": float(self.episode_env_reward),
            "episode_raw_reward": float(self.episode_raw_reward),
            "episode_unclipped_reward": float(self.episode_unclipped_reward),
            "episode_clipped_reward": float(self.episode_clipped_reward),
            "last_loss": float(self.last_loss),
            "last_policy_loss": float(self.last_policy_loss),
            "last_value_loss": float(self.last_value_loss),
            "last_entropy": float(self.last_entropy),
            "last_approximate_kl": float(self.last_approximate_kl),
            "last_clip_fraction": float(self.last_clip_fraction),
            "training_updates": int(self.training_updates),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore explicit rollout and metric counter state from a checkpoint."""
        state = checkpoint.get("mario_rl_state", {})
        self.env_frames = int(state.get("env_frames", self.env_frames))
        self.episodes = int(state.get("episodes", self.episodes))
        self.episode_reward = float(state.get("episode_reward", self.episode_reward))
        self.episode_env_reward = float(
            state.get("episode_env_reward", self.episode_env_reward)
        )
        self.episode_raw_reward = float(
            state.get("episode_raw_reward", self.episode_raw_reward)
        )
        self.episode_unclipped_reward = float(
            state.get("episode_unclipped_reward", self.episode_unclipped_reward)
        )
        self.episode_clipped_reward = float(
            state.get("episode_clipped_reward", self.episode_clipped_reward)
        )
        self.last_loss = float(state.get("last_loss", self.last_loss))
        self.last_policy_loss = float(
            state.get("last_policy_loss", self.last_policy_loss)
        )
        self.last_value_loss = float(state.get("last_value_loss", self.last_value_loss))
        self.last_entropy = float(state.get("last_entropy", self.last_entropy))
        self.last_approximate_kl = float(
            state.get("last_approximate_kl", self.last_approximate_kl)
        )
        self.last_clip_fraction = float(
            state.get("last_clip_fraction", self.last_clip_fraction)
        )
        self.training_updates = int(state.get("training_updates", self.training_updates))

    def close_env(self) -> None:
        """Close the active environment if one has been created."""
        if self.env is not None:
            self.env.close()
            self.env = None
            self._active_task_env_id = None
            self._last_state = None
            self._hidden_state = None

    def metrics_summary(self) -> dict[str, Any]:
        """Return stable final metrics for command-level artifact writing."""
        metrics_payload = self.metrics_payload(include_active=True)
        global_metrics = metrics_payload["global"]
        return {
            "global_step": int(self.global_step),
            "env_frames": int(self.env_frames),
            "episodes": int(self.episodes),
            "episode_reward": float(self.episode_reward),
            "episode_env_reward": float(self.episode_env_reward),
            "episode_raw_reward": float(self.episode_raw_reward),
            "episode_unclipped_reward": float(self.episode_unclipped_reward),
            "episode_clipped_reward": float(self.episode_clipped_reward),
            "epsilon": 0.0,
            "loss": float(self.last_loss),
            "learning_rate": float(self.config.model.learning_rate),
            "metric_episode_count": int(global_metrics["episode_count"]),
            "metric_completed_episode_count": int(
                global_metrics["completed_episode_count"]
            ),
            "metric_step_count": int(global_metrics["step_count"]),
            "metric_frame_count": int(global_metrics["frame_count"]),
            "episode_return_total": float(global_metrics["episode_return_total"]),
            "transformed_return_total": float(
                global_metrics["transformed_return_total"]
            ),
            "clear_count": int(global_metrics["clear_count"]),
            "clear_rate": float(global_metrics["clear_rate"] or 0.0),
            "death_count": int(global_metrics["death_count"]),
            "death_rate": float(global_metrics["death_rate"] or 0.0),
            "timeout_count": int(global_metrics["timeout_count"]),
            "truncation_count": int(global_metrics["truncation_count"]),
            "max_progress": float(global_metrics["max_progress"] or 0.0),
            "final_progress_mean": float(global_metrics["final_progress_mean"] or 0.0),
            "algorithm": "ppo",
            "ppo_policy_loss": float(self.last_policy_loss),
            "ppo_value_loss": float(self.last_value_loss),
            "ppo_entropy": float(self.last_entropy),
            "ppo_approximate_kl": float(self.last_approximate_kl),
            "ppo_clip_fraction": float(self.last_clip_fraction),
            "metrics_payload": metrics_payload,
        }

    def metrics_payload(self, *, include_active: bool = False) -> dict[str, Any]:
        """Return the structured Mario metrics payload for artifacts."""
        return self.metrics.to_payload(include_active=include_active)

    def _ensure_env(self) -> None:
        if (
            self.env is not None
            and self._last_state is not None
            and self._hidden_state is not None
        ):
            return
        self._reset_active_episode()

    def _reset_active_episode(self) -> None:
        task = self._task_for_current_episode()
        env_id = task.env_id if task is not None else self.config.env.id
        if self.env is not None and self._active_task_env_id != env_id:
            self.env.close()
            self.env = None
            self._last_state = None
        if self.env is None:
            active_config = self._config_for_env_id(env_id)
            if self._env_factory is None:
                from mario_rl.envs import make_env

                self.env = make_env(config=active_config.env.to_mario_env_config())
            else:
                self.env = self._env_factory(active_config)
            self._active_task_env_id = env_id
            self._set_task_features_for_env_id(env_id)
        assert self.env is not None
        state, reset_info = self.env.reset(seed=self.config.env.seed)
        self.metrics.start_episode(
            reset_info if isinstance(reset_info, dict) else None,
            fallback_env_id=env_id,
        )
        self._last_state = self._coerce_state(state)
        self._hidden_state = self.policy.initial_state(1, device=self.device)
        self.episode_reward = 0.0
        self.episode_env_reward = 0.0
        self.episode_raw_reward = 0.0
        self.episode_unclipped_reward = 0.0
        self.episode_clipped_reward = 0.0

    def _task_for_current_episode(self):
        if self.task_suite is None:
            return None
        return self.task_suite.task_for_episode(self.episodes)

    def _config_for_env_id(self, env_id: str) -> MarioRLConfig:
        if env_id == self.config.env.id:
            return self.config
        return replace(self.config, env=replace(self.config.env, id=env_id))

    def _collect_rollout(self) -> RolloutStorage:
        rollout = self._new_rollout_storage()
        while not rollout.full:
            assert self.env is not None
            assert self._last_state is not None
            assert self._hidden_state is not None
            observation = self._last_state
            hidden_before = self._hidden_state.detach()
            state_tensor = torch.as_tensor(
                observation,
                device=self.device,
            ).unsqueeze(0)
            with torch.no_grad():
                output = self.policy(
                    state_tensor,
                    hidden_before,
                    self._task_feature_tensor(batch_size=1),
                )
                distribution = Categorical(logits=output.policy_logits)
                action_tensor = distribution.sample()
                log_probability = distribution.log_prob(action_tensor)
                value = output.value

            action = int(action_tensor.detach().cpu().item())
            next_state, reward, terminated, truncated, info = self.env.step(action)
            transformed = self.reward_transformer.transform(
                float(reward),
                info if isinstance(info, dict) else None,
            )
            next_state = self._coerce_state(next_state)
            info_map = info if isinstance(info, dict) else None
            frames = int(info.get("frames_skipped", 1)) if isinstance(info, dict) else 1
            rollout.insert(
                observation,
                action,
                float(log_probability.detach().cpu().item()),
                transformed.training_reward,
                bool(terminated),
                bool(truncated),
                float(value.detach().cpu().item()),
                hidden_before.detach().cpu().numpy(),
                task_features=self._task_features,
                env_reward=transformed.env_reward,
                raw_reward=transformed.raw_reward,
                unclipped_reward=transformed.unclipped_reward,
                clipped_reward=transformed.clipped_reward,
                frames_skipped=max(frames, 1),
            )
            self.metrics.observe_step(
                reward=float(reward),
                transformed=transformed,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info_map,
                fallback_env_id=self._active_task_env_id,
                frame_count=max(frames, 1),
            )
            self.env_frames += max(frames, 1)
            self.episode_reward += transformed.training_reward
            self.episode_env_reward += transformed.env_reward
            self.episode_raw_reward += transformed.raw_reward
            if transformed.unclipped_reward is not None:
                self.episode_unclipped_reward += transformed.unclipped_reward
            if transformed.clipped_reward is not None:
                self.episode_clipped_reward += transformed.clipped_reward

            self._hidden_state = output.hidden_state.detach()
            if terminated or truncated:
                self._hidden_state = self.policy.reset_recurrent_state(
                    self._hidden_state,
                    torch.ones(1, dtype=torch.bool, device=self.device),
                )
                self.metrics.finish_episode(
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                self.episodes += 1
                self._reset_active_episode()
            else:
                self._last_state = next_state
        return rollout

    def _next_value(self) -> np.ndarray:
        assert self._last_state is not None
        assert self._hidden_state is not None
        state_tensor = torch.as_tensor(
            self._last_state,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            output = self.policy(
                state_tensor,
                self._hidden_state,
                self._task_feature_tensor(batch_size=1),
            )
        return output.value.detach().cpu().numpy().reshape(1)

    def _optimize_rollout(self, rollout: RolloutStorage) -> dict[str, torch.Tensor]:
        optimizer = self.optimizers()
        totals = {
            "total": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
        }
        count = 0
        for _ in range(int(self.config.ppo.epochs)):
            for batch in rollout.minibatches(
                int(self.config.ppo.minibatch_size),
                device=self.device,
            ):
                hidden_state = batch.hidden_state.permute(1, 0, 2).contiguous()
                output = self.policy(
                    batch.observation,
                    hidden_state,
                    batch.task_features,
                )
                loss = compute_ppo_loss(
                    output.policy_logits,
                    output.value,
                    batch.action,
                    batch.old_log_probability,
                    batch.return_,
                    batch.advantage,
                    clip_range=self.config.ppo.clip_range,
                    value_loss_coefficient=self.config.ppo.value_loss_coefficient,
                    entropy_coefficient=self.config.ppo.entropy_coefficient,
                    normalize_advantages=self.config.ppo.normalize_advantages,
                )
                optimizer.zero_grad()
                self.manual_backward(loss.total)
                if self.config.ppo.max_grad_norm is not None:
                    max_norm = float(self.config.ppo.max_grad_norm)
                    if max_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm)
                optimizer.step()
                self.training_updates += 1
                totals["total"] += float(loss.total.detach().cpu().item())
                totals["policy"] += float(loss.policy.detach().cpu().item())
                totals["value"] += float(loss.value.detach().cpu().item())
                totals["entropy"] += float(loss.entropy.detach().cpu().item())
                totals["approximate_kl"] += float(
                    loss.approximate_kl.detach().cpu().item()
                )
                totals["clip_fraction"] += float(loss.clip_fraction.detach().cpu().item())
                count += 1
        if count == 0:
            raise RuntimeError("PPO rollout produced no minibatches")
        return {
            name: torch.as_tensor(total / count, dtype=torch.float32, device=self.device)
            for name, total in totals.items()
        }

    def _new_rollout_storage(self) -> RolloutStorage:
        task_feature_shape = None
        if int(self.policy.task_feature_size) > 0:
            task_feature_shape = (int(self.policy.task_feature_size),)
        return RolloutStorage(
            rollout_steps=int(self.config.ppo.rollout_steps),
            num_envs=1,
            observation_shape=tuple(self.config.replay.state_shape),
            observation_dtype=np.dtype(self.config.replay.sample_dtype),
            hidden_state_shape=(1, int(self.policy.recurrent_hidden_size)),
            task_feature_shape=task_feature_shape,
            seed=(
                self.config.trainer.seed
                if self.config.trainer.seed is not None
                else self.config.env.seed
            ),
        )

    def _configure_task_features(self) -> None:
        feature_size = int(getattr(self.policy, "task_feature_size", 0))
        if feature_size <= 0:
            return
        self.task_encoder = TaskFeatureEncoder()
        if self.task_encoder.feature_size != feature_size:
            raise ValueError(
                "configured task feature size "
                f"{feature_size} does not match encoder size {self.task_encoder.feature_size}"
            )
        self._set_task_features_for_env_id(self.config.env.id)

    def _set_task_features_for_env_id(self, env_id: str) -> None:
        if self.task_encoder is None:
            return
        self._task_features = self.task_encoder.encode_env_id(env_id).vector

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


__all__ = ["PPOLightningModule"]
