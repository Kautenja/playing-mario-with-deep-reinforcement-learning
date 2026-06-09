"""Exploration schedules and action-selection helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LinearEpsilonSchedule:
    """Linearly anneal epsilon from ``start`` to ``final`` over frames."""

    start: float = 1.0
    final: float = 0.1
    decay_frames: int = 1_000_000
    current_step: int = 0

    def value(self, step: int | None = None) -> float:
        """Return the epsilon value at ``step`` or at the current schedule step."""
        step = self.current_step if step is None else int(step)
        if self.decay_frames <= 0:
            return float(self.final)
        if step >= self.decay_frames:
            return float(self.final)
        ratio = min(max(step, 0) / int(self.decay_frames), 1.0)
        return float(self.start + ratio * (self.final - self.start))

    def step(self, amount: int = 1) -> float:
        """Advance the schedule and return the new epsilon value."""
        self.current_step += int(amount)
        return self.value()

    def state_dict(self) -> dict[str, float | int]:
        """Return checkpoint-friendly primitive schedule state."""
        return {
            "start": float(self.start),
            "final": float(self.final),
            "decay_frames": int(self.decay_frames),
            "current_step": int(self.current_step),
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        """Restore schedule state from :meth:`state_dict`."""
        self.start = float(state["start"])
        self.final = float(state["final"])
        self.decay_frames = int(state["decay_frames"])
        self.current_step = int(state["current_step"])


class EpsilonGreedyActionSelector:
    """Seedable epsilon-greedy action selector."""

    def __init__(self, *, num_actions: int, seed: int | None = None) -> None:
        self.num_actions = int(num_actions)
        if self.num_actions <= 0:
            raise ValueError("num_actions must be > 0")
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        q_values: torch.Tensor,
        *,
        epsilon: float,
        deterministic: bool = False,
    ) -> int:
        """Select an action from one vector of Q-values."""
        q_values = torch.as_tensor(q_values)
        if q_values.ndim != 1:
            raise ValueError(f"q_values must be one-dimensional, got {tuple(q_values.shape)}")
        if q_values.shape[0] != self.num_actions:
            raise ValueError("q_values length must equal num_actions")
        if deterministic or float(epsilon) <= 0.0 or self.rng.random() >= float(epsilon):
            return int(q_values.argmax().item())
        return int(self.rng.integers(self.num_actions))


__all__ = ["EpsilonGreedyActionSelector", "LinearEpsilonSchedule"]
