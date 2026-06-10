"""On-policy rollout storage for recurrent actor-critic training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutBatch:
    """Torch minibatch sampled from a completed rollout."""

    observation: torch.Tensor
    action: torch.Tensor
    old_log_probability: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    value: torch.Tensor
    return_: torch.Tensor
    advantage: torch.Tensor
    hidden_state: torch.Tensor
    task_features: torch.Tensor | None = None
    env_reward: torch.Tensor | None = None
    raw_reward: torch.Tensor | None = None
    unclipped_reward: torch.Tensor | None = None
    clipped_reward: torch.Tensor | None = None
    frames_skipped: torch.Tensor | None = None
    auxiliary_targets: dict[str, torch.Tensor] | None = None
    auxiliary_masks: dict[str, torch.Tensor] | None = None


class RolloutStorage:
    """Fixed-size rollout buffer with generalized advantage estimation."""

    def __init__(
        self,
        *,
        rollout_steps: int,
        num_envs: int,
        observation_shape: tuple[int, ...],
        hidden_state_shape: tuple[int, int],
        observation_dtype: np.dtype | str = np.uint8,
        task_feature_shape: tuple[int, ...] | None = None,
        auxiliary_target_names: tuple[str, ...] = (),
        seed: int | None = None,
    ) -> None:
        self.rollout_steps = _positive_int(rollout_steps, "rollout_steps")
        self.num_envs = _positive_int(num_envs, "num_envs")
        self.observation_shape = tuple(int(dimension) for dimension in observation_shape)
        self.hidden_state_shape = tuple(int(dimension) for dimension in hidden_state_shape)
        if len(self.hidden_state_shape) != 2:
            raise ValueError("hidden_state_shape must be (layers, hidden_size)")
        self.task_feature_shape = (
            tuple(int(dimension) for dimension in task_feature_shape)
            if task_feature_shape is not None
            else None
        )
        self.auxiliary_target_names = tuple(str(name) for name in auxiliary_target_names)
        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._advantages_ready = False

        shape = (self.rollout_steps, self.num_envs)
        self.observations = np.empty(
            (*shape, *self.observation_shape),
            dtype=np.dtype(observation_dtype),
        )
        self.actions = np.empty(shape, dtype=np.int64)
        self.old_log_probabilities = np.empty(shape, dtype=np.float32)
        self.rewards = np.empty(shape, dtype=np.float32)
        self.terminated = np.empty(shape, dtype=np.bool_)
        self.truncated = np.empty(shape, dtype=np.bool_)
        self.values = np.empty(shape, dtype=np.float32)
        self.returns = np.empty(shape, dtype=np.float32)
        self.advantages = np.empty(shape, dtype=np.float32)
        self.hidden_states = np.empty(
            (*shape, *self.hidden_state_shape),
            dtype=np.float32,
        )
        self.task_features = (
            np.empty((*shape, *self.task_feature_shape), dtype=np.float32)
            if self.task_feature_shape is not None
            else None
        )
        self.env_rewards = np.empty(shape, dtype=np.float32)
        self.raw_rewards = np.empty(shape, dtype=np.float32)
        self.unclipped_rewards = np.empty(shape, dtype=np.float32)
        self.clipped_rewards = np.empty(shape, dtype=np.float32)
        self.frames_skipped = np.ones(shape, dtype=np.int32)
        self.auxiliary_targets = {
            name: np.zeros(shape, dtype=np.float32)
            for name in self.auxiliary_target_names
        }
        self.auxiliary_masks = {
            name: np.zeros(shape, dtype=np.bool_)
            for name in self.auxiliary_target_names
        }

    def __len__(self) -> int:
        """Return the number of inserted rollout steps."""
        return self._position

    @property
    def full(self) -> bool:
        """Return whether the rollout has all configured steps populated."""
        return self._position >= self.rollout_steps

    def reset(self) -> None:
        """Clear inserted rollout steps while retaining allocated arrays."""
        self._position = 0
        self._advantages_ready = False

    def insert(
        self,
        observation: np.ndarray,
        action: int | np.ndarray,
        old_log_probability: float | np.ndarray,
        reward: float | np.ndarray,
        terminated: bool | np.ndarray,
        truncated: bool | np.ndarray,
        value: float | np.ndarray,
        hidden_state: np.ndarray,
        *,
        task_features: np.ndarray | None = None,
        env_reward: float | np.ndarray | None = None,
        raw_reward: float | np.ndarray | None = None,
        unclipped_reward: float | np.ndarray | None = None,
        clipped_reward: float | np.ndarray | None = None,
        frames_skipped: int | np.ndarray = 1,
        auxiliary_targets: dict[str, float | np.ndarray] | None = None,
        auxiliary_masks: dict[str, bool | np.ndarray] | None = None,
    ) -> None:
        """Insert one vectorized step at the next rollout position."""
        if self.full:
            raise ValueError("rollout storage is full")
        index = self._position
        self.observations[index] = self._coerce_observation(observation)
        self.actions[index] = np.asarray(action, dtype=np.int64).reshape(self.num_envs)
        self.old_log_probabilities[index] = _float_row(
            old_log_probability,
            self.num_envs,
        )
        self.rewards[index] = _float_row(reward, self.num_envs)
        self.terminated[index] = _bool_row(terminated, self.num_envs)
        self.truncated[index] = _bool_row(truncated, self.num_envs)
        self.values[index] = _float_row(value, self.num_envs)
        self.hidden_states[index] = self._coerce_hidden_state(hidden_state)
        if self.task_features is not None:
            self.task_features[index] = self._coerce_task_features(task_features)
        elif task_features is not None:
            raise ValueError("task_features require task_feature_shape")
        self.env_rewards[index] = _float_row(env_reward, self.num_envs, default=reward)
        self.raw_rewards[index] = _float_row(raw_reward, self.num_envs, default=reward)
        self.unclipped_rewards[index] = _float_row(unclipped_reward, self.num_envs)
        self.clipped_rewards[index] = _float_row(clipped_reward, self.num_envs)
        self.frames_skipped[index] = np.asarray(frames_skipped, dtype=np.int32).reshape(
            self.num_envs
        )
        self._insert_auxiliary_targets(
            index,
            auxiliary_targets=auxiliary_targets,
            auxiliary_masks=auxiliary_masks,
        )
        self._position += 1
        self._advantages_ready = False

    def compute_returns_and_advantages(
        self,
        next_values: np.ndarray | torch.Tensor,
        *,
        discount_factor: float,
        gae_lambda: float,
    ) -> None:
        """Populate GAE advantages and bootstrapped returns for a full rollout."""
        if not self.full:
            raise ValueError("cannot compute advantages before rollout is full")
        next_value = np.asarray(next_values, dtype=np.float32).reshape(self.num_envs)
        gae = np.zeros(self.num_envs, dtype=np.float32)
        discount_factor = float(discount_factor)
        gae_lambda = float(gae_lambda)
        for step in range(self.rollout_steps - 1, -1, -1):
            done = self.terminated[step] | self.truncated[step]
            nonterminal = (~done).astype(np.float32)
            delta = (
                self.rewards[step]
                + discount_factor * next_value * nonterminal
                - self.values[step]
            )
            gae = delta + discount_factor * gae_lambda * nonterminal * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
            next_value = self.values[step]
        self._advantages_ready = True

    def minibatches(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        shuffle: bool = True,
    ):
        """Yield flattened minibatches from a completed rollout."""
        if not self._advantages_ready:
            raise ValueError("compute_returns_and_advantages must run first")
        batch_size = _positive_int(batch_size, "batch_size")
        total = self.rollout_steps * self.num_envs
        indices = np.arange(total)
        if shuffle:
            self._rng.shuffle(indices)
        for start in range(0, total, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield RolloutBatch(
                observation=_tensor(self.observations.reshape(total, *self.observation_shape)[batch_indices], device=device),
                action=_long_tensor(self.actions.reshape(total)[batch_indices], device=device),
                old_log_probability=_tensor(self.old_log_probabilities.reshape(total)[batch_indices], device=device),
                reward=_tensor(self.rewards.reshape(total)[batch_indices], device=device),
                terminated=_bool_tensor(self.terminated.reshape(total)[batch_indices], device=device),
                truncated=_bool_tensor(self.truncated.reshape(total)[batch_indices], device=device),
                value=_tensor(self.values.reshape(total)[batch_indices], device=device),
                return_=_tensor(self.returns.reshape(total)[batch_indices], device=device),
                advantage=_tensor(self.advantages.reshape(total)[batch_indices], device=device),
                hidden_state=_tensor(
                    self.hidden_states.reshape(total, *self.hidden_state_shape)[batch_indices],
                    device=device,
                ),
                task_features=(
                    _tensor(
                        self.task_features.reshape(total, *self.task_feature_shape)[batch_indices],
                        device=device,
                    )
                    if self.task_features is not None
                    else None
                ),
                env_reward=_tensor(self.env_rewards.reshape(total)[batch_indices], device=device),
                raw_reward=_tensor(self.raw_rewards.reshape(total)[batch_indices], device=device),
                unclipped_reward=_tensor(
                    self.unclipped_rewards.reshape(total)[batch_indices],
                    device=device,
                ),
                clipped_reward=_tensor(
                    self.clipped_rewards.reshape(total)[batch_indices],
                    device=device,
                ),
                frames_skipped=_long_tensor(
                    self.frames_skipped.reshape(total)[batch_indices],
                    device=device,
                ),
                auxiliary_targets=(
                    {
                        name: _tensor(values.reshape(total)[batch_indices], device=device)
                        for name, values in self.auxiliary_targets.items()
                    }
                    if self.auxiliary_target_names
                    else None
                ),
                auxiliary_masks=(
                    {
                        name: _bool_tensor(values.reshape(total)[batch_indices], device=device)
                        for name, values in self.auxiliary_masks.items()
                    }
                    if self.auxiliary_target_names
                    else None
                ),
            )

    def _coerce_observation(self, observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation, dtype=self.observations.dtype)
        if array.shape == self.observation_shape:
            array = array.reshape((1, *self.observation_shape))
        if array.shape != (self.num_envs, *self.observation_shape):
            raise ValueError(
                f"expected observation shape {(self.num_envs, *self.observation_shape)} "
                f"or {self.observation_shape}, got {array.shape}"
            )
        return array

    def _coerce_hidden_state(self, hidden_state: np.ndarray) -> np.ndarray:
        array = np.asarray(hidden_state, dtype=np.float32)
        layers, hidden_size = self.hidden_state_shape
        if array.shape == (layers, self.num_envs, hidden_size):
            array = np.transpose(array, (1, 0, 2))
        if array.shape == self.hidden_state_shape and self.num_envs == 1:
            array = array.reshape((1, *self.hidden_state_shape))
        if array.shape != (self.num_envs, *self.hidden_state_shape):
            raise ValueError(
                f"expected hidden state shape {(self.num_envs, *self.hidden_state_shape)} "
                f"or {(layers, self.num_envs, hidden_size)}, got {array.shape}"
            )
        return array

    def _coerce_task_features(self, task_features: np.ndarray | None) -> np.ndarray:
        if self.task_feature_shape is None:
            raise ValueError("task_feature_shape is not configured")
        if task_features is None:
            raise ValueError("task_features are required for this rollout")
        array = np.asarray(task_features, dtype=np.float32)
        if array.shape == self.task_feature_shape:
            array = array.reshape((1, *self.task_feature_shape))
        if array.shape != (self.num_envs, *self.task_feature_shape):
            raise ValueError(
                f"expected task feature shape {(self.num_envs, *self.task_feature_shape)} "
                f"or {self.task_feature_shape}, got {array.shape}"
            )
        return array

    def _insert_auxiliary_targets(
        self,
        index: int,
        *,
        auxiliary_targets: dict[str, float | np.ndarray] | None,
        auxiliary_masks: dict[str, bool | np.ndarray] | None,
    ) -> None:
        if not self.auxiliary_target_names:
            if auxiliary_targets or auxiliary_masks:
                raise ValueError("auxiliary targets require auxiliary_target_names")
            return
        auxiliary_targets = auxiliary_targets or {}
        auxiliary_masks = auxiliary_masks or {}
        unknown = (set(auxiliary_targets) | set(auxiliary_masks)) - set(
            self.auxiliary_target_names
        )
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown auxiliary target(s): {names}")
        for name in self.auxiliary_target_names:
            value = auxiliary_targets.get(name, 0.0)
            mask = auxiliary_masks.get(name, False)
            self.auxiliary_targets[name][index] = _float_row(value, self.num_envs)
            self.auxiliary_masks[name][index] = _bool_row(mask, self.num_envs)


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _float_row(
    value,
    num_envs: int,
    *,
    default=None,
) -> np.ndarray:
    if value is None:
        value = default
    if value is None:
        return np.full(num_envs, np.nan, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(num_envs)


def _bool_row(value, num_envs: int) -> np.ndarray:
    return np.asarray(value, dtype=np.bool_).reshape(num_envs)


def _tensor(value, *, device=None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _long_tensor(value, *, device=None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long, device=device)


def _bool_tensor(value, *, device=None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.bool, device=device)
