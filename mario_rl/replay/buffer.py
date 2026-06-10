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
    task_features: torch.Tensor | None = None
    next_task_features: torch.Tensor | None = None


@dataclass(frozen=True)
class ReplayBatch:
    """NumPy replay batch with stable field names, shapes, and dtypes."""

    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_state: np.ndarray
    task_features: np.ndarray | None = None
    next_task_features: np.ndarray | None = None

    def to_torch(self, device: torch.device | str | None = None) -> TorchReplayBatch:
        """Return a tensor batch, moving arrays to ``device`` only at this boundary."""
        return TorchReplayBatch(
            state=torch.as_tensor(self.state, device=device),
            action=torch.as_tensor(self.action, dtype=torch.long, device=device),
            reward=torch.as_tensor(self.reward, dtype=torch.float32, device=device),
            terminated=torch.as_tensor(self.terminated, dtype=torch.bool, device=device),
            truncated=torch.as_tensor(self.truncated, dtype=torch.bool, device=device),
            next_state=torch.as_tensor(self.next_state, device=device),
            task_features=_optional_tensor(self.task_features, device=device),
            next_task_features=_optional_tensor(self.next_task_features, device=device),
        )


class UniformReplayBuffer:
    """Fixed-capacity uniform replay buffer backed by typed NumPy arrays."""

    def __init__(
        self,
        *,
        capacity: int,
        state_shape: tuple[int, ...],
        state_dtype: np.dtype | str = np.uint8,
        task_feature_shape: tuple[int, ...] | None = None,
        task_feature_dtype: np.dtype | str = np.float32,
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
        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0
        self._states = np.empty((capacity, *self.state_shape), dtype=self.state_dtype)
        self._next_states = np.empty((capacity, *self.state_shape), dtype=self.state_dtype)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.bool_)
        self._truncated = np.empty(capacity, dtype=np.bool_)
        self._task_features = None
        self._next_task_features = None
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
        replace = self._size < batch_size
        indices = self._rng.choice(self._size, size=batch_size, replace=replace)
        batch = ReplayBatch(
            state=self._states[indices],
            action=self._actions[indices],
            reward=self._rewards[indices],
            terminated=self._terminated[indices],
            truncated=self._truncated[indices],
            next_state=self._next_states[indices],
            task_features=(
                self._task_features[indices] if self._task_features is not None else None
            ),
            next_task_features=(
                self._next_task_features[indices]
                if self._next_task_features is not None
                else None
            ),
        )
        if as_tensors or device is not None:
            return batch.to_torch(device=device)
        return batch

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


def build_replay_buffer(config: Any, *, seed: int | None = None) -> UniformReplayBuffer:
    """Build the active replay buffer from a typed config object."""
    replay_config = getattr(config, "replay", config)
    model_config = getattr(config, "model", None)
    if bool(getattr(replay_config, "prioritized", False)):
        raise NotImplementedError(
            "prioritized replay is not part of the active PyTorch path yet"
        )
    task_feature_shape = None
    if bool(getattr(model_config, "task_conditioning", False)):
        feature_size = int(getattr(model_config, "task_feature_size", 0) or 0)
        if feature_size <= 0:
            from mario_rl.envs import task_feature_size

            feature_size = int(task_feature_size())
        task_feature_shape = (feature_size,)
    return UniformReplayBuffer(
        capacity=int(getattr(replay_config, "capacity")),
        state_shape=tuple(getattr(replay_config, "state_shape")),
        state_dtype=np.dtype(getattr(replay_config, "sample_dtype", np.uint8)),
        task_feature_shape=task_feature_shape,
        seed=seed,
    )


def _optional_tensor(
    value: np.ndarray | None,
    *,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.float32, device=device)
