"""Replay-buffer implementations with explicit batch contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TorchReplayBatch:
    """Torch tensor replay batch used at training boundaries."""

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_state: torch.Tensor
    env_reward: torch.Tensor | None = None
    raw_reward: torch.Tensor | None = None
    unclipped_reward: torch.Tensor | None = None
    clipped_reward: torch.Tensor | None = None
    task_features: torch.Tensor | None = None
    next_task_features: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    importance_weights: torch.Tensor | None = None


@dataclass(frozen=True)
class ReplayBatch:
    """NumPy replay batch with stable field names, shapes, and dtypes."""

    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_state: np.ndarray
    env_reward: np.ndarray | None = None
    raw_reward: np.ndarray | None = None
    unclipped_reward: np.ndarray | None = None
    clipped_reward: np.ndarray | None = None
    task_features: np.ndarray | None = None
    next_task_features: np.ndarray | None = None
    indices: np.ndarray | None = None
    importance_weights: np.ndarray | None = None

    def to_torch(self, device: torch.device | str | None = None) -> TorchReplayBatch:
        """Return a tensor batch, moving arrays to ``device`` only at this boundary."""
        return TorchReplayBatch(
            state=torch.as_tensor(self.state, device=device),
            action=torch.as_tensor(self.action, dtype=torch.long, device=device),
            reward=torch.as_tensor(self.reward, dtype=torch.float32, device=device),
            terminated=torch.as_tensor(self.terminated, dtype=torch.bool, device=device),
            truncated=torch.as_tensor(self.truncated, dtype=torch.bool, device=device),
            next_state=torch.as_tensor(self.next_state, device=device),
            env_reward=_optional_float_tensor(self.env_reward, device=device),
            raw_reward=_optional_float_tensor(self.raw_reward, device=device),
            unclipped_reward=_optional_float_tensor(
                self.unclipped_reward,
                device=device,
            ),
            clipped_reward=_optional_float_tensor(self.clipped_reward, device=device),
            task_features=_optional_tensor(self.task_features, device=device),
            next_task_features=_optional_tensor(self.next_task_features, device=device),
            indices=_optional_long_tensor(self.indices, device=device),
            importance_weights=_optional_float_tensor(
                self.importance_weights,
                device=device,
            ),
        )


class UniformReplayBuffer:
    """Fixed-capacity uniform replay buffer backed by typed NumPy arrays."""

    prioritized = False

    def __init__(
        self,
        *,
        capacity: int,
        state_shape: tuple[int, ...],
        state_dtype: np.dtype | str = np.uint8,
        task_feature_shape: tuple[int, ...] | None = None,
        task_feature_dtype: np.dtype | str = np.float32,
        store_reward_info: bool = False,
        seed: int | None = None,
    ) -> None:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.state_shape = tuple(int(dimension) for dimension in state_shape)
        self.state_dtype = np.dtype(state_dtype)
        self.task_feature_shape = (
            tuple(int(dimension) for dimension in task_feature_shape)
            if task_feature_shape is not None
            else None
        )
        self.task_feature_dtype = np.dtype(task_feature_dtype)
        self.store_reward_info = bool(store_reward_info)
        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0
        self._states = np.empty((capacity, *self.state_shape), dtype=self.state_dtype)
        self._next_states = np.empty((capacity, *self.state_shape), dtype=self.state_dtype)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.bool_)
        self._truncated = np.empty(capacity, dtype=np.bool_)
        self._env_rewards = None
        self._raw_rewards = None
        self._unclipped_rewards = None
        self._clipped_rewards = None
        self._task_features = None
        self._next_task_features = None
        if self.store_reward_info:
            self._env_rewards = np.empty(capacity, dtype=np.float32)
            self._raw_rewards = np.empty(capacity, dtype=np.float32)
            self._unclipped_rewards = np.empty(capacity, dtype=np.float32)
            self._clipped_rewards = np.empty(capacity, dtype=np.float32)
        if self.task_feature_shape is not None:
            self._task_features = np.empty(
                (capacity, *self.task_feature_shape),
                dtype=self.task_feature_dtype,
            )
            self._next_task_features = np.empty(
                (capacity, *self.task_feature_shape),
                dtype=self.task_feature_dtype,
            )

    def __len__(self) -> int:
        """Return the number of populated transitions."""
        return self._size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_state: np.ndarray,
        *,
        env_reward: float | None = None,
        raw_reward: float | None = None,
        unclipped_reward: float | None = None,
        clipped_reward: float | None = None,
        task_features: np.ndarray | None = None,
        next_task_features: np.ndarray | None = None,
    ) -> None:
        """Insert one transition, overwriting the oldest item at capacity."""
        self._states[self._position] = self._coerce_state(state)
        self._actions[self._position] = int(action)
        self._rewards[self._position] = float(reward)
        self._terminated[self._position] = bool(terminated)
        self._truncated[self._position] = bool(truncated)
        self._next_states[self._position] = self._coerce_state(next_state)
        if self.store_reward_info:
            if (
                self._env_rewards is None
                or self._raw_rewards is None
                or self._unclipped_rewards is None
                or self._clipped_rewards is None
            ):
                raise AssertionError("reward info arrays were not initialized")
            self._env_rewards[self._position] = _optional_reward_value(
                env_reward,
                default=reward,
            )
            self._raw_rewards[self._position] = _optional_reward_value(
                raw_reward,
                default=reward,
            )
            self._unclipped_rewards[self._position] = _optional_reward_value(
                unclipped_reward,
            )
            self._clipped_rewards[self._position] = _optional_reward_value(
                clipped_reward,
            )
        if self.task_feature_shape is not None:
            if self._task_features is None or self._next_task_features is None:
                raise AssertionError("task feature arrays were not initialized")
            self._task_features[self._position] = self._coerce_task_features(task_features)
            if next_task_features is None:
                next_task_features = task_features
            self._next_task_features[self._position] = self._coerce_task_features(
                next_task_features
            )
        elif task_features is not None or next_task_features is not None:
            raise ValueError("task_features require task_feature_shape")
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        as_tensors: bool = False,
    ) -> ReplayBatch | TorchReplayBatch:
        """Sample a batch uniformly, using replacement when the buffer is small."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = self._sample_uniform_indices(batch_size)
        batch = self._batch_for_indices(indices)
        if as_tensors or device is not None:
            return batch.to_torch(device=device)
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Reject priority updates for the uniform buffer contract."""
        raise RuntimeError("uniform replay buffer does not track priorities")

    def priority_summary(self) -> dict[str, Any]:
        """Return JSON-safe replay sampling metadata."""
        return {
            "prioritized": False,
            "capacity": int(self.capacity),
            "size": int(self._size),
            "priority_alpha": None,
            "priority_beta": None,
            "priority_epsilon": None,
            "priority_updates": 0,
            "max_priority": None,
            "mean_priority": None,
        }

    def _sample_uniform_indices(self, batch_size: int) -> np.ndarray:
        batch_size = self._validate_batch_size(batch_size)
        replace = self._size < batch_size
        return self._rng.choice(self._size, size=batch_size, replace=replace)

    def _validate_batch_size(self, batch_size: int) -> int:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        return batch_size

    def _batch_for_indices(
        self,
        indices: np.ndarray,
        *,
        sample_indices: np.ndarray | None = None,
        importance_weights: np.ndarray | None = None,
    ) -> ReplayBatch:
        indices = np.asarray(indices, dtype=np.int64)
        return ReplayBatch(
            state=self._states[indices],
            action=self._actions[indices],
            reward=self._rewards[indices],
            terminated=self._terminated[indices],
            truncated=self._truncated[indices],
            next_state=self._next_states[indices],
            env_reward=(
                self._env_rewards[indices] if self._env_rewards is not None else None
            ),
            raw_reward=(
                self._raw_rewards[indices] if self._raw_rewards is not None else None
            ),
            unclipped_reward=(
                self._unclipped_rewards[indices]
                if self._unclipped_rewards is not None
                else None
            ),
            clipped_reward=(
                self._clipped_rewards[indices]
                if self._clipped_rewards is not None
                else None
            ),
            task_features=(
                self._task_features[indices] if self._task_features is not None else None
            ),
            next_task_features=(
                self._next_task_features[indices]
                if self._next_task_features is not None
                else None
            ),
            indices=sample_indices,
            importance_weights=importance_weights,
        )

    def _coerce_state(self, state: np.ndarray) -> np.ndarray:
        array = np.asarray(state, dtype=self.state_dtype)
        if array.shape != self.state_shape:
            raise ValueError(f"expected state shape {self.state_shape}, got {array.shape}")
        return array

    def _coerce_task_features(self, task_features: np.ndarray | None) -> np.ndarray:
        if self.task_feature_shape is None:
            raise ValueError("task_feature_shape is not configured")
        if task_features is None:
            raise ValueError("task_features are required for this replay buffer")
        array = np.asarray(task_features, dtype=self.task_feature_dtype)
        if array.shape != self.task_feature_shape:
            raise ValueError(
                f"expected task feature shape {self.task_feature_shape}, got {array.shape}"
            )
        return array


class PrioritizedReplayBuffer(UniformReplayBuffer):
    """Fixed-capacity proportional prioritized replay buffer."""

    prioritized = True

    def __init__(
        self,
        *,
        capacity: int,
        state_shape: tuple[int, ...],
        state_dtype: np.dtype | str = np.uint8,
        task_feature_shape: tuple[int, ...] | None = None,
        task_feature_dtype: np.dtype | str = np.float32,
        store_reward_info: bool = False,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        priority_epsilon: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        self.priority_alpha = float(priority_alpha)
        self.priority_beta = float(priority_beta)
        self.priority_epsilon = float(priority_epsilon)
        if self.priority_alpha < 0.0:
            raise ValueError("priority_alpha must be >= 0")
        if self.priority_beta < 0.0:
            raise ValueError("priority_beta must be >= 0")
        if self.priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be > 0")
        super().__init__(
            capacity=capacity,
            state_shape=state_shape,
            state_dtype=state_dtype,
            task_feature_shape=task_feature_shape,
            task_feature_dtype=task_feature_dtype,
            store_reward_info=store_reward_info,
            seed=seed,
        )
        self._priorities = np.zeros(self.capacity, dtype=np.float32)
        self._max_priority = 1.0
        self.priority_updates = 0

    def push(self, *args, **kwargs) -> None:
        """Insert one transition with the current maximum priority."""
        index = self._position
        super().push(*args, **kwargs)
        self._priorities[index] = max(self._max_priority, self.priority_epsilon)

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        as_tensors: bool = False,
    ) -> ReplayBatch | TorchReplayBatch:
        """Sample according to proportional priorities with IS weights."""
        batch_size = self._validate_batch_size(batch_size)
        probabilities = self._sampling_probabilities()
        replace = self._size < batch_size
        indices = self._rng.choice(
            self._size,
            size=batch_size,
            replace=replace,
            p=probabilities,
        )
        weights = self._importance_weights(indices, probabilities)
        batch = self._batch_for_indices(
            indices,
            sample_indices=np.asarray(indices, dtype=np.int64),
            importance_weights=weights,
        )
        if as_tensors or device is not None:
            return batch.to_torch(device=device)
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update sampled transition priorities in-place."""
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        priority_array = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if index_array.shape != priority_array.shape:
            raise ValueError("indices and priorities must have the same shape")
        if index_array.size == 0:
            return
        if np.any(index_array < 0) or np.any(index_array >= self._size):
            raise IndexError("priority indices must refer to populated transitions")
        if not np.all(np.isfinite(priority_array)):
            raise ValueError("priorities must be finite")
        priority_array = np.maximum(priority_array, self.priority_epsilon)
        self._priorities[index_array] = priority_array
        self._max_priority = max(self._max_priority, float(priority_array.max()))
        self.priority_updates += int(index_array.size)

    def priority_summary(self) -> dict[str, Any]:
        """Return JSON-safe prioritized replay metadata."""
        active = self._priorities[: self._size]
        populated = active[active > 0.0]
        return {
            "prioritized": True,
            "capacity": int(self.capacity),
            "size": int(self._size),
            "priority_alpha": float(self.priority_alpha),
            "priority_beta": float(self.priority_beta),
            "priority_epsilon": float(self.priority_epsilon),
            "priority_updates": int(self.priority_updates),
            "max_priority": (
                float(populated.max()) if populated.size else float(self._max_priority)
            ),
            "mean_priority": (
                float(populated.mean()) if populated.size else float(self._max_priority)
            ),
        }

    def _sampling_probabilities(self) -> np.ndarray:
        active = np.asarray(self._priorities[: self._size], dtype=np.float64)
        if self.priority_alpha == 0.0:
            return np.full(self._size, 1.0 / self._size, dtype=np.float64)
        scaled = np.power(np.maximum(active, self.priority_epsilon), self.priority_alpha)
        total = float(scaled.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full(self._size, 1.0 / self._size, dtype=np.float64)
        return scaled / total

    def _importance_weights(
        self,
        indices: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        if self.priority_beta == 0.0:
            return np.ones(len(indices), dtype=np.float32)
        selected = np.asarray(probabilities[indices], dtype=np.float64)
        weights = np.power(self._size * selected, -self.priority_beta)
        maximum = float(weights.max()) if weights.size else 1.0
        if maximum > 0.0 and np.isfinite(maximum):
            weights = weights / maximum
        return weights.astype(np.float32)


def build_replay_buffer(config: Any, *, seed: int | None = None) -> UniformReplayBuffer:
    """Build the active replay buffer from a typed config object."""
    replay_config = getattr(config, "replay", config)
    model_config = getattr(config, "model", None)
    task_feature_shape = None
    if bool(getattr(model_config, "task_conditioning", False)):
        feature_size = int(getattr(model_config, "task_feature_size", 0) or 0)
        if feature_size <= 0:
            from mario_rl.envs import task_feature_size

            feature_size = int(task_feature_size())
        task_feature_shape = (feature_size,)
    buffer_type = (
        PrioritizedReplayBuffer
        if bool(getattr(replay_config, "prioritized", False))
        else UniformReplayBuffer
    )
    kwargs: dict[str, Any] = {}
    if buffer_type is PrioritizedReplayBuffer:
        kwargs.update(
            priority_alpha=float(getattr(replay_config, "priority_alpha", 0.6)),
            priority_beta=float(getattr(replay_config, "priority_beta", 0.4)),
            priority_epsilon=float(getattr(replay_config, "priority_epsilon", 1e-6)),
        )
    return buffer_type(
        capacity=int(getattr(replay_config, "capacity")),
        state_shape=tuple(getattr(replay_config, "state_shape")),
        state_dtype=np.dtype(getattr(replay_config, "sample_dtype", np.uint8)),
        task_feature_shape=task_feature_shape,
        store_reward_info=bool(getattr(replay_config, "store_reward_info", False)),
        seed=seed,
        **kwargs,
    )


def _optional_tensor(
    value: np.ndarray | None,
    *,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _optional_float_tensor(
    value: np.ndarray | None,
    *,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _optional_long_tensor(
    value: np.ndarray | None,
    *,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.long, device=device)


def _optional_reward_value(
    value: float | None,
    *,
    default: float | None = None,
) -> float:
    if value is None:
        value = default
    if value is None:
        return float("nan")
    return float(value)
