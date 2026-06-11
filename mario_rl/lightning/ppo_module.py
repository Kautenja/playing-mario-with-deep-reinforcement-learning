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
from mario_rl.auxiliary import (
    auxiliary_loss_weights,
    auxiliary_target_names,
    extract_auxiliary_targets,
)
from mario_rl.config import MarioRLConfig, to_dict, with_resolved_model_num_actions
from mario_rl.envs import TaskFeatureEncoder, build_task_sampler
from mario_rl.exploration import RandomNetworkDistillation
from mario_rl.lightning.data import build_step_dataloader
from mario_rl.metrics import MarioMetricsAccumulator
from mario_rl.models import (
    AuxiliaryLoss,
    RecurrentActorCritic,
    build_model,
    compute_auxiliary_loss,
    compute_ppo_loss,
    make_optimizer,
)
from mario_rl.rewards import RewardTransformer
from mario_rl.snapshots import SnapshotLibrary


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
        self.num_envs = _positive_int(self.config.ppo.num_envs, "ppo.num_envs")

        policy = build_model(self.config)
        if not isinstance(policy, RecurrentActorCritic):
            raise TypeError(
                "PPOLightningModule requires model.architecture='recurrent_actor_critic'"
            )
        self.policy = policy
        self.rnd: RandomNetworkDistillation | None = None
        if bool(self.config.exploration.enabled):
            self.rnd = RandomNetworkDistillation(
                input_shape=tuple(self.config.replay.state_shape),
                embedding_size=int(self.config.exploration.rnd_embedding_size),
                hidden_size=int(self.config.exploration.rnd_hidden_size),
                normalize_observations=bool(
                    self.config.exploration.normalize_observations
                ),
                normalize_intrinsic_rewards=bool(
                    self.config.exploration.normalize_intrinsic_rewards
                ),
                intrinsic_reward_clip=self.config.exploration.intrinsic_reward_clip,
            )

        self.reward_transformer = RewardTransformer(self.config.reward_transform)
        self.metrics = MarioMetricsAccumulator(default_task_id=self.config.env.id)
        self.task_suite = (
            build_task_sampler(self.config.task_suite)
            if bool(getattr(self.config.task_suite, "enabled", False))
            else None
        )
        seed = (
            self.config.trainer.seed
            if self.config.trainer.seed is not None
            else self.config.env.seed
        )
        self.snapshot_library = SnapshotLibrary(self.config.snapshot, seed=seed)
        self.task_encoder = None
        self._task_features: np.ndarray | None = None
        self._configure_task_features()
        self._auxiliary_target_names = auxiliary_target_names(self.config.auxiliary)
        self._auxiliary_loss_weights = (
            auxiliary_loss_weights(self.config.auxiliary)
            if self._auxiliary_target_names
            else {}
        )

        self.env = None
        self.envs: list[Any | None] = []
        self._active_task_env_ids: list[str | None] = []
        self._last_states: list[np.ndarray | None] = []
        self._slot_episode_indices: list[int | None] = []
        self._next_episode_index = 0
        self._hidden_state: torch.Tensor | None = None
        self.env_frames = 0
        self.episodes = 0
        self._slot_episode_reward = [0.0 for _ in range(self.num_envs)]
        self._slot_episode_transformed_reward = [0.0 for _ in range(self.num_envs)]
        self._slot_episode_intrinsic_reward = [0.0 for _ in range(self.num_envs)]
        self._slot_episode_env_reward = [0.0 for _ in range(self.num_envs)]
        self._slot_episode_raw_reward = [0.0 for _ in range(self.num_envs)]
        self._slot_episode_unclipped_reward = [0.0 for _ in range(self.num_envs)]
        self._slot_episode_clipped_reward = [0.0 for _ in range(self.num_envs)]
        self.episode_reward = 0.0
        self.episode_transformed_reward = 0.0
        self.episode_intrinsic_reward = 0.0
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
        self.last_auxiliary_loss = 0.0
        self.last_auxiliary_losses = {
            target: 0.0 for target in self._auxiliary_target_names
        }
        self.last_auxiliary_valid_counts = {
            target: 0.0 for target in self._auxiliary_target_names
        }
        self.last_intrinsic_reward_total = 0.0
        self.last_intrinsic_reward_mean = 0.0
        self.last_rnd_raw_error_mean = 0.0
        self.last_rnd_loss = 0.0
        self.last_rnd_predictor_grad_norm = 0.0
        self.training_updates = 0

    def train_dataloader(self):
        """Use one placeholder item per PPO rollout/update cycle."""
        return build_step_dataloader(self.config.train.max_steps)

    def configure_optimizers(self):
        """Create the configured optimizer for the actor-critic policy."""
        policy_optimizer = make_optimizer(self.policy, self.config)
        if self.rnd is None:
            return policy_optimizer
        rnd_optimizer = torch.optim.Adam(
            self.rnd.predictor.parameters(),
            lr=float(self.config.exploration.predictor_learning_rate),
        )
        return [policy_optimizer, rnd_optimizer]

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
        self.last_auxiliary_loss = float(losses["auxiliary"].detach().cpu().item())
        for target in self._auxiliary_target_names:
            self.last_auxiliary_losses[target] = float(
                losses[f"auxiliary/{target}"].detach().cpu().item()
            )
            self.last_auxiliary_valid_counts[target] = float(
                losses[f"auxiliary_valid/{target}"].detach().cpu().item()
            )

        self.log("train/loss", losses["total"], on_step=True, prog_bar=False)
        self.log("train/ppo_loss", losses["ppo_total"], on_step=True, prog_bar=False)
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
        self.log(
            "train/ppo_num_envs",
            float(self.num_envs),
            on_step=True,
            prog_bar=False,
        )
        if bool(self.config.exploration.log_intrinsic_rewards):
            self.log(
                "train/intrinsic_reward_total",
                self.last_intrinsic_reward_total,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                "train/intrinsic_reward_mean",
                self.last_intrinsic_reward_mean,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                "train/rnd_raw_error_mean",
                self.last_rnd_raw_error_mean,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                "train/rnd_loss",
                self.last_rnd_loss,
                on_step=True,
                prog_bar=False,
            )
            self.log(
                "train/rnd_predictor_grad_norm",
                self.last_rnd_predictor_grad_norm,
                on_step=True,
                prog_bar=False,
            )
        self.log(
            "train/auxiliary_loss",
            losses["auxiliary"],
            on_step=True,
            prog_bar=False,
        )
        for target in self._auxiliary_target_names:
            self.log(
                f"train/auxiliary_{target}_loss",
                losses[f"auxiliary/{target}"],
                on_step=True,
                prog_bar=False,
            )
            self.log(
                f"train/auxiliary_{target}_valid",
                losses[f"auxiliary_valid/{target}"],
                on_step=True,
                prog_bar=False,
            )
        self.log("train/episode_reward", self.episode_reward, on_step=True, prog_bar=False)
        self.log(
            "train/episode_transformed_reward",
            self.episode_transformed_reward,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            "train/episode_intrinsic_reward",
            self.episode_intrinsic_reward,
            on_step=True,
            prog_bar=False,
        )
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
            "num_envs": int(self.num_envs),
            "env_frames": int(self.env_frames),
            "episodes": int(self.episodes),
            "next_episode_index": int(self._next_episode_index),
            "episode_reward": float(self.episode_reward),
            "episode_transformed_reward": float(self.episode_transformed_reward),
            "episode_intrinsic_reward": float(self.episode_intrinsic_reward),
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
            "last_auxiliary_loss": float(self.last_auxiliary_loss),
            "last_auxiliary_losses": dict(self.last_auxiliary_losses),
            "last_auxiliary_valid_counts": dict(self.last_auxiliary_valid_counts),
            "last_intrinsic_reward_total": float(self.last_intrinsic_reward_total),
            "last_intrinsic_reward_mean": float(self.last_intrinsic_reward_mean),
            "last_rnd_raw_error_mean": float(self.last_rnd_raw_error_mean),
            "last_rnd_loss": float(self.last_rnd_loss),
            "last_rnd_predictor_grad_norm": float(
                self.last_rnd_predictor_grad_norm
            ),
            "training_updates": int(self.training_updates),
        }
        if self.task_suite is not None and hasattr(self.task_suite, "state_dict"):
            checkpoint["mario_rl_task_suite_state"] = self.task_suite.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore explicit rollout and metric counter state from a checkpoint."""
        state = checkpoint.get("mario_rl_state", {})
        saved_num_envs = int(state.get("num_envs", self.num_envs))
        if saved_num_envs != self.num_envs:
            raise ValueError(
                f"checkpoint PPO num_envs={saved_num_envs} does not match "
                f"configured num_envs={self.num_envs}"
            )
        self.env_frames = int(state.get("env_frames", self.env_frames))
        self.episodes = int(state.get("episodes", self.episodes))
        self._next_episode_index = int(
            state.get("next_episode_index", max(self._next_episode_index, self.episodes))
        )
        self.episode_reward = float(state.get("episode_reward", self.episode_reward))
        self.episode_transformed_reward = float(
            state.get("episode_transformed_reward", self.episode_transformed_reward)
        )
        self.episode_intrinsic_reward = float(
            state.get("episode_intrinsic_reward", self.episode_intrinsic_reward)
        )
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
        self.last_auxiliary_loss = float(
            state.get("last_auxiliary_loss", self.last_auxiliary_loss)
        )
        self.last_auxiliary_losses.update(
            {
                str(name): float(value)
                for name, value in dict(state.get("last_auxiliary_losses", {})).items()
                if str(name) in self.last_auxiliary_losses
            }
        )
        self.last_auxiliary_valid_counts.update(
            {
                str(name): float(value)
                for name, value in dict(
                    state.get("last_auxiliary_valid_counts", {})
                ).items()
                if str(name) in self.last_auxiliary_valid_counts
            }
        )
        self.last_intrinsic_reward_total = float(
            state.get("last_intrinsic_reward_total", self.last_intrinsic_reward_total)
        )
        self.last_intrinsic_reward_mean = float(
            state.get("last_intrinsic_reward_mean", self.last_intrinsic_reward_mean)
        )
        self.last_rnd_raw_error_mean = float(
            state.get("last_rnd_raw_error_mean", self.last_rnd_raw_error_mean)
        )
        self.last_rnd_loss = float(state.get("last_rnd_loss", self.last_rnd_loss))
        self.last_rnd_predictor_grad_norm = float(
            state.get(
                "last_rnd_predictor_grad_norm",
                self.last_rnd_predictor_grad_norm,
            )
        )
        self.training_updates = int(state.get("training_updates", self.training_updates))
        if self.task_suite is not None and "mario_rl_task_suite_state" in checkpoint:
            self.task_suite.load_state_dict(checkpoint["mario_rl_task_suite_state"])

    def close_env(self) -> None:
        """Close all active rollout environments."""
        for env in self.envs:
            if env is not None:
                env.close()
        self.env = None
        self.envs = []
        self._active_task_env_ids = []
        self._last_states = []
        self._slot_episode_indices = []
        self._hidden_state = None

    def metrics_summary(self) -> dict[str, Any]:
        """Return stable final metrics for command-level artifact writing."""
        metrics_payload = self.metrics_payload(include_active=True)
        global_metrics = metrics_payload["global"]
        return {
            "global_step": int(self.global_step),
            "env_frames": int(self.env_frames),
            "episodes": int(self.episodes),
            "ppo_num_envs": int(self.num_envs),
            "episode_reward": float(self.episode_reward),
            "episode_transformed_reward": float(self.episode_transformed_reward),
            "episode_intrinsic_reward": float(self.episode_intrinsic_reward),
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
            "exploration_enabled": bool(self.config.exploration.enabled),
            "intrinsic_reward_total": float(self.last_intrinsic_reward_total),
            "intrinsic_reward_mean": float(self.last_intrinsic_reward_mean),
            "rnd_raw_error_mean": float(self.last_rnd_raw_error_mean),
            "rnd_loss": float(self.last_rnd_loss),
            "rnd_predictor_grad_norm": float(self.last_rnd_predictor_grad_norm),
            "auxiliary_loss": float(self.last_auxiliary_loss),
            "auxiliary_losses": dict(self.last_auxiliary_losses),
            "auxiliary_valid_counts": dict(self.last_auxiliary_valid_counts),
            "metrics_payload": metrics_payload,
            **self._task_suite_metric_counts(),
        }

    def metrics_payload(self, *, include_active: bool = False) -> dict[str, Any]:
        """Return the structured Mario metrics payload for artifacts."""
        return self.metrics.to_payload(include_active=include_active)

    def task_suite_payload(self) -> dict[str, Any] | None:
        """Return sampler metadata/state for artifacts."""
        if self.task_suite is None:
            return None
        if hasattr(self.task_suite, "payload"):
            return self.task_suite.payload()
        return {
            "metadata": self.task_suite.metadata(),
            "state": self.task_suite.state_dict(),
            "counts": self.task_suite.summary_counts(),
        }

    def snapshot_payload(self) -> dict[str, Any] | None:
        """Return JSON-safe snapshot metadata for training artifacts."""
        if not self.snapshot_library.enabled and not self.snapshot_library.entries:
            return None
        return self.snapshot_library.payload()

    def _ensure_env(self) -> None:
        if self._vector_state_ready():
            return
        self._initialize_vector_state()

    def _vector_state_ready(self) -> bool:
        return (
            len(self.envs) == self.num_envs
            and len(self._last_states) == self.num_envs
            and all(env is not None for env in self.envs)
            and all(state is not None for state in self._last_states)
            and self._hidden_state is not None
            and tuple(self._hidden_state.shape)
            == (1, self.num_envs, int(self.policy.recurrent_hidden_size))
        )

    def _initialize_vector_state(self) -> None:
        if len(self.envs) != self.num_envs:
            self.close_env()
            self.envs = [None for _ in range(self.num_envs)]
            self._active_task_env_ids = [None for _ in range(self.num_envs)]
            self._last_states = [None for _ in range(self.num_envs)]
            self._slot_episode_indices = [None for _ in range(self.num_envs)]
        self._hidden_state = self.policy.initial_state(self.num_envs, device=self.device)
        self._ensure_task_feature_matrix()
        for slot in range(self.num_envs):
            self._reset_slot(slot)

    def _reset_slot(self, slot: int) -> None:
        slot = int(slot)
        episode_index = self._next_episode_index
        self._next_episode_index += 1
        self._slot_episode_indices[slot] = episode_index
        task = self._task_for_episode(episode_index)
        env_id = task.env_id if task is not None else self.config.env.id
        env = self.envs[slot]
        if env is not None and self._active_task_env_ids[slot] != env_id:
            env.close()
            self.envs[slot] = None
            self._last_states[slot] = None
            env = None
        if env is None:
            active_config = self._config_for_env_id(env_id)
            if self._env_factory is None:
                from mario_rl.envs import make_env

                env = make_env(config=active_config.env.to_mario_env_config())
            else:
                env = self._env_factory(active_config)
            self.envs[slot] = env
            self._active_task_env_ids[slot] = env_id
            self.env = self.envs[0]
        self._set_task_features_for_slot(slot, env_id)
        assert env is not None
        seed = self._seed_for_episode(episode_index)
        state, reset_info = self.snapshot_library.reset_or_restore(
            env,
            env_id=env_id,
            action_set=self._action_set_for_env(env),
            seed=seed,
        )
        self.metrics.start_episode(
            reset_info if isinstance(reset_info, dict) else None,
            fallback_env_id=env_id,
            slot=slot,
        )
        self._last_states[slot] = self._coerce_state(state)
        self._slot_episode_reward[slot] = 0.0
        self._slot_episode_transformed_reward[slot] = 0.0
        self._slot_episode_intrinsic_reward[slot] = 0.0
        self._slot_episode_env_reward[slot] = 0.0
        self._slot_episode_raw_reward[slot] = 0.0
        self._slot_episode_unclipped_reward[slot] = 0.0
        self._slot_episode_clipped_reward[slot] = 0.0
        self._sync_episode_totals()

    def _task_for_episode(self, episode_index: int):
        if self.task_suite is None:
            return None
        return self.task_suite.task_for_episode(int(episode_index))

    def _seed_for_episode(self, episode_index: int) -> int | None:
        if self.config.env.seed is None:
            return None
        return int(self.config.env.seed) + int(episode_index)

    def _config_for_env_id(self, env_id: str) -> MarioRLConfig:
        if env_id == self.config.env.id:
            return self.config
        return replace(self.config, env=replace(self.config.env, id=env_id))

    def _collect_rollout(self) -> RolloutStorage:
        rollout = self._new_rollout_storage()
        while not rollout.full:
            assert self._hidden_state is not None
            observation = self._stack_observations()
            hidden_before = self._hidden_state.detach()
            state_tensor = torch.as_tensor(
                observation,
                device=self.device,
            )
            with torch.no_grad():
                output = self.policy(
                    state_tensor,
                    hidden_before,
                    self._task_feature_tensor(),
                )
                distribution = Categorical(logits=output.policy_logits)
                action_tensor = distribution.sample()
                log_probability = distribution.log_prob(action_tensor)
                value = output.value

            actions = action_tensor.detach().cpu().numpy().astype(np.int64)
            log_probabilities = log_probability.detach().cpu().numpy().astype(np.float32)
            values = value.detach().cpu().numpy().astype(np.float32)
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            transformed_rewards = np.zeros(self.num_envs, dtype=np.float32)
            intrinsic_rewards = np.zeros(self.num_envs, dtype=np.float32)
            env_rewards = np.zeros(self.num_envs, dtype=np.float32)
            raw_rewards = np.zeros(self.num_envs, dtype=np.float32)
            unclipped_rewards = np.full(self.num_envs, np.nan, dtype=np.float32)
            clipped_rewards = np.full(self.num_envs, np.nan, dtype=np.float32)
            terminated_flags = np.zeros(self.num_envs, dtype=np.bool_)
            truncated_flags = np.zeros(self.num_envs, dtype=np.bool_)
            frames_skipped = np.ones(self.num_envs, dtype=np.int32)
            auxiliary_values = {
                name: np.zeros(self.num_envs, dtype=np.float32)
                for name in self._auxiliary_target_names
            }
            auxiliary_masks = {
                name: np.zeros(self.num_envs, dtype=np.bool_)
                for name in self._auxiliary_target_names
            }
            next_states = []

            for slot in range(self.num_envs):
                env = self.envs[slot]
                assert env is not None
                next_state, reward, terminated, truncated, info = env.step(
                    int(actions[slot])
                )
                transformed = self.reward_transformer.transform(
                    float(reward),
                    info if isinstance(info, dict) else None,
                )
                next_state = self._coerce_state(next_state)
                info_map = info if isinstance(info, dict) else None
                auxiliary_targets = extract_auxiliary_targets(
                    info_map,
                    transformed_reward=transformed.training_reward,
                    targets=self._auxiliary_target_names,
                )
                frames = (
                    int(info.get("frames_skipped", 1))
                    if isinstance(info, dict)
                    else 1
                )
                frames = max(frames, 1)
                transformed_rewards[slot] = transformed.training_reward
                env_rewards[slot] = transformed.env_reward
                raw_rewards[slot] = transformed.raw_reward
                if transformed.unclipped_reward is not None:
                    unclipped_rewards[slot] = transformed.unclipped_reward
                if transformed.clipped_reward is not None:
                    clipped_rewards[slot] = transformed.clipped_reward
                terminated_flags[slot] = bool(terminated)
                truncated_flags[slot] = bool(truncated)
                frames_skipped[slot] = frames
                for name in self._auxiliary_target_names:
                    if name in auxiliary_targets.values:
                        auxiliary_values[name][slot] = float(auxiliary_targets.values[name])
                    if name in auxiliary_targets.masks:
                        auxiliary_masks[name][slot] = bool(auxiliary_targets.masks[name])
                self.metrics.observe_step(
                    reward=float(reward),
                    transformed=transformed,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info_map,
                    fallback_env_id=self._active_task_env_ids[slot],
                    frame_count=frames,
                    slot=slot,
                )
                self.env_frames += frames
                self._slot_episode_transformed_reward[slot] += (
                    transformed.training_reward
                )
                self._slot_episode_env_reward[slot] += transformed.env_reward
                self._slot_episode_raw_reward[slot] += transformed.raw_reward
                if transformed.unclipped_reward is not None:
                    self._slot_episode_unclipped_reward[slot] += (
                        transformed.unclipped_reward
                    )
                if transformed.clipped_reward is not None:
                    self._slot_episode_clipped_reward[slot] += transformed.clipped_reward
                self._last_states[slot] = next_state
                next_states.append(next_state)
                self._maybe_capture_snapshot(
                    slot,
                    next_state,
                    info_map,
                    terminal=bool(terminated or truncated),
                )

            if next_states:
                intrinsic_rewards = self._intrinsic_rewards(np.stack(next_states, axis=0))
            rewards = transformed_rewards + intrinsic_rewards
            for slot in range(self.num_envs):
                self._slot_episode_reward[slot] += float(rewards[slot])
                self._slot_episode_intrinsic_reward[slot] += float(
                    intrinsic_rewards[slot]
                )

            rollout.insert(
                observation,
                actions,
                log_probabilities,
                rewards,
                terminated_flags,
                truncated_flags,
                values,
                hidden_before.detach().cpu().numpy(),
                task_features=self._task_features,
                env_reward=env_rewards,
                raw_reward=raw_rewards,
                transformed_reward=transformed_rewards,
                intrinsic_reward=intrinsic_rewards,
                unclipped_reward=unclipped_rewards,
                clipped_reward=clipped_rewards,
                frames_skipped=frames_skipped,
                auxiliary_targets=auxiliary_values,
                auxiliary_masks=auxiliary_masks,
            )
            self._hidden_state = output.hidden_state.detach()
            done = terminated_flags | truncated_flags
            if np.any(done):
                self._hidden_state = self.policy.reset_recurrent_state(
                    self._hidden_state,
                    torch.as_tensor(done, dtype=torch.bool, device=self.device),
                )
                for slot, is_done in enumerate(done):
                    if not bool(is_done):
                        continue
                    episode_metrics = self.metrics.finish_episode(
                        terminated=bool(terminated_flags[slot]),
                        truncated=bool(truncated_flags[slot]),
                        slot=slot,
                    )
                    if episode_metrics is not None and self.task_suite is not None:
                        self.task_suite.observe_episode(
                            episode_metrics,
                            env_id=self._active_task_env_ids[slot],
                        )
                    self.episodes += 1
                    self._reset_slot(slot)
            self._sync_episode_totals()
        return rollout

    def _next_value(self) -> np.ndarray:
        assert self._hidden_state is not None
        state_tensor = torch.as_tensor(
            self._stack_observations(),
            device=self.device,
        )
        with torch.no_grad():
            output = self.policy(
                state_tensor,
                self._hidden_state,
                self._task_feature_tensor(),
            )
        return output.value.detach().cpu().numpy().reshape(self.num_envs)

    def _optimize_rollout(self, rollout: RolloutStorage) -> dict[str, torch.Tensor]:
        optimizer = self._policy_optimizer()
        totals = {
            "total": 0.0,
            "ppo_total": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "auxiliary": 0.0,
        }
        for target in self._auxiliary_target_names:
            totals[f"auxiliary/{target}"] = 0.0
            totals[f"auxiliary_valid/{target}"] = 0.0
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
                if self._auxiliary_loss_weights:
                    auxiliary_loss = compute_auxiliary_loss(
                        output.auxiliary,
                        batch.auxiliary_targets,
                        batch.auxiliary_masks,
                        weights=self._auxiliary_loss_weights,
                    )
                else:
                    auxiliary_loss = AuxiliaryLoss(
                        total=loss.total * 0.0,
                        terms={},
                        weighted_terms={},
                        valid_counts={},
                    )
                total_loss = loss.total + auxiliary_loss.total
                optimizer.zero_grad()
                self.manual_backward(total_loss)
                if self.config.ppo.max_grad_norm is not None:
                    max_norm = float(self.config.ppo.max_grad_norm)
                    if max_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm)
                optimizer.step()
                self.training_updates += 1
                totals["total"] += float(total_loss.detach().cpu().item())
                totals["ppo_total"] += float(loss.total.detach().cpu().item())
                totals["policy"] += float(loss.policy.detach().cpu().item())
                totals["value"] += float(loss.value.detach().cpu().item())
                totals["entropy"] += float(loss.entropy.detach().cpu().item())
                totals["approximate_kl"] += float(
                    loss.approximate_kl.detach().cpu().item()
                )
                totals["clip_fraction"] += float(loss.clip_fraction.detach().cpu().item())
                totals["auxiliary"] += float(
                    auxiliary_loss.total.detach().cpu().item()
                )
                for target in self._auxiliary_target_names:
                    totals[f"auxiliary/{target}"] += float(
                        auxiliary_loss.terms[target].detach().cpu().item()
                    )
                    totals[f"auxiliary_valid/{target}"] += float(
                        auxiliary_loss.valid_counts[target].detach().cpu().item()
                    )
                count += 1
        if count == 0:
            raise RuntimeError("PPO rollout produced no minibatches")
        return {
            name: torch.as_tensor(total / count, dtype=torch.float32, device=self.device)
            for name, total in totals.items()
        }

    def _intrinsic_rewards(self, next_observations: np.ndarray) -> np.ndarray:
        if self.rnd is None:
            self.last_intrinsic_reward_total = 0.0
            self.last_intrinsic_reward_mean = 0.0
            self.last_rnd_raw_error_mean = 0.0
            self.last_rnd_loss = 0.0
            self.last_rnd_predictor_grad_norm = 0.0
            return np.zeros(self.num_envs, dtype=np.float32)
        reward = self.rnd(torch.as_tensor(next_observations, device=self.device))
        scale = self._active_intrinsic_reward_scale()
        scaled = reward.intrinsic_reward * scale
        optimizer = self._rnd_optimizer()
        if optimizer is not None:
            optimizer.zero_grad()
            self.manual_backward(reward.predictor_loss)
            self.last_rnd_predictor_grad_norm = self._predictor_grad_norm()
            optimizer.step()
        else:
            self.last_rnd_predictor_grad_norm = 0.0
        intrinsic = scaled.detach().cpu().numpy().astype(np.float32).reshape(self.num_envs)
        self.last_intrinsic_reward_total = float(np.sum(intrinsic))
        self.last_intrinsic_reward_mean = float(np.mean(intrinsic))
        self.last_rnd_raw_error_mean = float(
            reward.raw_error.detach().mean().cpu().item()
        )
        self.last_rnd_loss = float(reward.predictor_loss.detach().cpu().item())
        return intrinsic

    def _active_intrinsic_reward_scale(self) -> float:
        if self.env_frames < int(self.config.exploration.warmup_steps):
            return float(self.config.exploration.warmup_reward_scale)
        return float(self.config.exploration.intrinsic_reward_scale)

    def _policy_optimizer(self):
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            return optimizers[0]
        return optimizers

    def _rnd_optimizer(self):
        if self.rnd is None:
            return None
        try:
            optimizers = self.optimizers(use_pl_optimizer=False)
        except RuntimeError:
            return None
        if isinstance(optimizers, (list, tuple)) and len(optimizers) > 1:
            return optimizers[1]
        return None

    def _predictor_grad_norm(self) -> float:
        if self.rnd is None:
            return 0.0
        total = 0.0
        for parameter in self.rnd.predictor.parameters():
            if parameter.grad is None:
                continue
            total += float(parameter.grad.detach().pow(2).sum().cpu().item())
        return float(total ** 0.5)

    def _new_rollout_storage(self) -> RolloutStorage:
        task_feature_shape = None
        if int(self.policy.task_feature_size) > 0:
            task_feature_shape = (int(self.policy.task_feature_size),)
        return RolloutStorage(
            rollout_steps=int(self.config.ppo.rollout_steps),
            num_envs=self.num_envs,
            observation_shape=tuple(self.config.replay.state_shape),
            observation_dtype=np.dtype(self.config.replay.sample_dtype),
            hidden_state_shape=(1, int(self.policy.recurrent_hidden_size)),
            task_feature_shape=task_feature_shape,
            auxiliary_target_names=self._auxiliary_target_names,
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
        self._ensure_task_feature_matrix()
        self._set_task_features_for_slot(0, self.config.env.id)

    def _ensure_task_feature_matrix(self) -> None:
        if self.task_encoder is None:
            return
        if (
            self._task_features is None
            or self._task_features.shape
            != (self.num_envs, int(self.policy.task_feature_size))
        ):
            self._task_features = np.zeros(
                (self.num_envs, int(self.policy.task_feature_size)),
                dtype=np.float32,
            )

    def _set_task_features_for_slot(self, slot: int, env_id: str) -> None:
        if self.task_encoder is None:
            return
        self._ensure_task_feature_matrix()
        assert self._task_features is not None
        self._task_features[int(slot)] = self.task_encoder.encode_env_id(env_id).vector

    def _task_feature_tensor(self) -> torch.Tensor | None:
        if self._task_features is None:
            return None
        return torch.as_tensor(
            self._task_features,
            dtype=torch.float32,
            device=self.device,
        )

    def _stack_observations(self) -> np.ndarray:
        if len(self._last_states) != self.num_envs:
            raise RuntimeError("PPO vector observations are not initialized")
        states = []
        for slot, state in enumerate(self._last_states):
            if state is None:
                raise RuntimeError(f"PPO vector slot {slot} has no observation")
            states.append(state)
        return np.stack(states, axis=0)

    def _sync_episode_totals(self) -> None:
        self.episode_reward = float(sum(self._slot_episode_reward))
        self.episode_transformed_reward = float(
            sum(self._slot_episode_transformed_reward)
        )
        self.episode_intrinsic_reward = float(
            sum(self._slot_episode_intrinsic_reward)
        )
        self.episode_env_reward = float(sum(self._slot_episode_env_reward))
        self.episode_raw_reward = float(sum(self._slot_episode_raw_reward))
        self.episode_unclipped_reward = float(
            sum(self._slot_episode_unclipped_reward)
        )
        self.episode_clipped_reward = float(sum(self._slot_episode_clipped_reward))

    def _coerce_state(self, state) -> np.ndarray:
        array = np.asarray(state, dtype=np.dtype(self.config.replay.sample_dtype))
        expected = tuple(self.config.replay.state_shape)
        if array.shape != expected:
            raise ValueError(f"expected observation shape {expected}, got {array.shape}")
        return array

    def _maybe_capture_snapshot(
        self,
        slot: int,
        observation: np.ndarray,
        info: dict[str, Any] | None,
        *,
        terminal: bool,
    ) -> None:
        slot = int(slot)
        env = self.envs[slot]
        if env is None:
            return
        active = self.metrics.active_episode(slot=slot)
        episode_step = active.step_count if active is not None else None
        episode_index = self._slot_episode_indices[slot]
        seed = (
            self._seed_for_episode(episode_index)
            if episode_index is not None
            else None
        )
        self.snapshot_library.maybe_capture(
            env,
            observation=observation,
            info=info,
            env_id=self._active_task_env_ids[slot] or self.config.env.id,
            action_set=self._action_set_for_env(env),
            seed_lineage=(seed, episode_index, slot),
            episode_step=episode_step,
            global_step=self.env_frames,
            terminal=terminal,
        )

    def _action_set_for_env(self, env: Any) -> str:
        return str(getattr(env, "mario_rl_action_set", self.config.env.action_set))

    def _task_suite_metric_counts(self) -> dict[str, Any]:
        if self.task_suite is None:
            return {
                "curriculum_mode": "disabled",
                "curriculum_active_count": 0,
                "curriculum_mastered_count": 0,
                "curriculum_locked_count": 0,
                "curriculum_retired_count": 0,
            }
        counts = self.task_suite.summary_counts()
        mode = getattr(getattr(self.task_suite, "config", None), "mode", "fixed")
        return {
            "curriculum_mode": str(mode),
            "curriculum_active_count": int(counts.get("active", 0)),
            "curriculum_mastered_count": int(counts.get("mastered", 0)),
            "curriculum_locked_count": int(counts.get("locked", 0)),
            "curriculum_retired_count": int(counts.get("retired", 0)),
        }


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


__all__ = ["PPOLightningModule"]
