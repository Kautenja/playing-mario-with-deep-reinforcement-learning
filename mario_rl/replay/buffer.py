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


@dataclass(frozen=True)
class ReplayBatch:
    """NumPy replay batch with stable field names, shapes, and dtypes."""

    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_state: np.ndarray

    def to_torch(self, device: torch.device | str | None = None) -> TorchReplayBatch:
        """Return a tensor batch, moving arrays to ``device`` only at this boundary."""
        return TorchReplayBatch(
            state=torch.as_tensor(self.state, device=device),
            action=torch.as_tensor(self.action, dtype=torch.long, device=device),
            reward=torch.as_tensor(self.reward, dtype=torch.float32, device=device),
            terminated=torch.as_tensor(self.terminated, dtype=torch.bool, device=device),
            truncated=torch.as_tensor(self.truncated, dtype=torch.bool, device=device),
            next_state=torch.as_tensor(self.next_state, device=device),
        )


class UniformReplayBuffer:
    """Fixed-capacity uniform replay buffer backed by typed NumPy arrays."""

    def __init__(
        self,
        *,
        capacity: int,
        state_shape: tuple[int, ...],
        state_dtype: np.dtype | str = np.uint8,
        seed: int | None = None,
    ) -> None:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.state_shape = tuple(int(dimension) for dimension in state_shape)
        self.state_dtype = np.dtype(state_dtype)
        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0
        self._states = np.empty((capacity, *self.state_shape), dtype=self.state_dtype)
        self._next_states = np.empty((capacity, *self.state_shape), dtype=self.state_dtype)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.bool_)
        self._truncated = np.empty(capacity, dtype=np.bool_)

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
    ) -> None:
        """Insert one transition, overwriting the oldest item at capacity."""
        self._states[self._position] = self._coerce_state(state)
        self._actions[self._position] = int(action)
        self._rewards[self._position] = float(reward)
        self._terminated[self._position] = bool(terminated)
        self._truncated[self._position] = bool(truncated)
        self._next_states[self._position] = self._coerce_state(next_state)
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
        )
        if as_tensors or device is not None:
            return batch.to_torch(device=device)
        return batch

    def _coerce_state(self, state: np.ndarray) -> np.ndarray:
        array = np.asarray(state, dtype=self.state_dtype)
        if array.shape != self.state_shape:
            raise ValueError(f"expected state shape {self.state_shape}, got {array.shape}")
        return array


def build_replay_buffer(config: Any, *, seed: int | None = None) -> UniformReplayBuffer:
    """Build the active replay buffer from a typed config object."""
    replay_config = getattr(config, "replay", config)
    if bool(getattr(replay_config, "prioritized", False)):
        raise NotImplementedError(
            "prioritized replay is not part of the active PyTorch path yet"
        )
    return UniformReplayBuffer(
        capacity=int(getattr(replay_config, "capacity")),
        state_shape=tuple(getattr(replay_config, "state_shape")),
        state_dtype=np.dtype(getattr(replay_config, "sample_dtype", np.uint8)),
        seed=seed,
    )
