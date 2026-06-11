"""Pixel-only exploration bonuses for Mario RL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

RND_OBSERVATION_SOURCE = "next_observation"


@dataclass(frozen=True)
class ExplorationConfig:
    """Optional intrinsic reward settings for pixel-only exploration."""

    enabled: bool = False
    method: str = "rnd"
    intrinsic_reward_scale: float = 0.05
    predictor_learning_rate: float = 0.0001
    normalize_observations: bool = True
    normalize_intrinsic_rewards: bool = True
    intrinsic_reward_clip: float | None = 1.0
    warmup_steps: int = 0
    warmup_reward_scale: float = 0.0
    log_intrinsic_rewards: bool = True
    rnd_embedding_size: int = 128
    rnd_hidden_size: int = 256
    observation_source: str = RND_OBSERVATION_SOURCE

    def __post_init__(self) -> None:
        method = str(self.method).lower()
        if method not in {"rnd"}:
            raise ValueError("exploration.method must be 'rnd'")
        if float(self.intrinsic_reward_scale) < 0.0:
            raise ValueError("exploration.intrinsic_reward_scale must be non-negative")
        if float(self.predictor_learning_rate) <= 0.0:
            raise ValueError("exploration.predictor_learning_rate must be > 0")
        if (
            self.intrinsic_reward_clip is not None
            and float(self.intrinsic_reward_clip) <= 0.0
        ):
            raise ValueError("exploration.intrinsic_reward_clip must be positive or null")
        if int(self.warmup_steps) < 0:
            raise ValueError("exploration.warmup_steps must be >= 0")
        if float(self.warmup_reward_scale) < 0.0:
            raise ValueError("exploration.warmup_reward_scale must be non-negative")
        if int(self.rnd_embedding_size) <= 0:
            raise ValueError("exploration.rnd_embedding_size must be > 0")
        if int(self.rnd_hidden_size) <= 0:
            raise ValueError("exploration.rnd_hidden_size must be > 0")
        if str(self.observation_source) != RND_OBSERVATION_SOURCE:
            raise ValueError(
                f"exploration.observation_source must be {RND_OBSERVATION_SOURCE!r}"
            )


@dataclass(frozen=True)
class RNDReward:
    """RND reward values and predictor loss for one observation batch."""

    intrinsic_reward: torch.Tensor
    raw_error: torch.Tensor
    normalized_error: torch.Tensor
    predictor_loss: torch.Tensor


class RandomNetworkDistillation(nn.Module):
    """Frozen-target RND module that consumes policy pixel observations only."""

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        embedding_size: int = 128,
        hidden_size: int = 256,
        normalize_observations: bool = True,
        normalize_intrinsic_rewards: bool = True,
        intrinsic_reward_clip: float | None = 1.0,
        input_scale: float = 255.0,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(dimension) for dimension in input_shape)
        if len(self.input_shape) != 3:
            raise ValueError(f"input_shape must be channel-first, got {input_shape!r}")
        self.normalize_observations = bool(normalize_observations)
        self.normalize_intrinsic_rewards = bool(normalize_intrinsic_rewards)
        self.intrinsic_reward_clip = (
            None
            if intrinsic_reward_clip is None
            else float(intrinsic_reward_clip)
        )
        self.input_scale = float(input_scale)
        self.target = _RNDNetwork(
            input_shape=self.input_shape,
            embedding_size=int(embedding_size),
            hidden_size=int(hidden_size),
        )
        self.predictor = _RNDNetwork(
            input_shape=self.input_shape,
            embedding_size=int(embedding_size),
            hidden_size=int(hidden_size),
        )
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.target.eval()
        self.register_buffer("_reward_mean", torch.zeros((), dtype=torch.float32))
        self.register_buffer("_reward_m2", torch.ones((), dtype=torch.float32))
        self.register_buffer("_reward_count", torch.zeros((), dtype=torch.float32))

    def forward(self, observation: torch.Tensor) -> RNDReward:
        """Return intrinsic rewards from next pixel observations."""
        flat = self._prepare_observation(observation)
        with torch.no_grad():
            target = self.target(flat)
        prediction = self.predictor(flat)
        raw_error = F.mse_loss(prediction, target, reduction="none").mean(dim=1)
        normalized = self._normalize_reward(raw_error.detach())
        intrinsic = normalized
        if self.intrinsic_reward_clip is not None:
            intrinsic = torch.clamp(intrinsic, min=0.0, max=self.intrinsic_reward_clip)
        predictor_loss = raw_error.mean()
        return RNDReward(
            intrinsic_reward=intrinsic,
            raw_error=raw_error.detach(),
            normalized_error=normalized.detach(),
            predictor_loss=predictor_loss,
        )

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(observation, device=self._reward_mean.device)
        if tensor.shape[-3:] != self.input_shape:
            raise ValueError(
                f"expected trailing observation shape {self.input_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        tensor = tensor.reshape(-1, *self.input_shape)
        if self.normalize_observations:
            return tensor.to(dtype=torch.float32) / self.input_scale
        return tensor.to(dtype=torch.float32)

    def _normalize_reward(self, raw_error: torch.Tensor) -> torch.Tensor:
        reward = raw_error.to(dtype=torch.float32)
        if not self.normalize_intrinsic_rewards:
            return reward
        variance = self._reward_variance()
        normalized = reward / torch.sqrt(variance + 1e-8)
        self._update_reward_statistics(reward)
        return normalized

    def _reward_variance(self) -> torch.Tensor:
        if float(self._reward_count.item()) < 2.0:
            return torch.ones_like(self._reward_m2)
        return self._reward_m2 / torch.clamp(self._reward_count - 1.0, min=1.0)

    def _update_reward_statistics(self, values: torch.Tensor) -> None:
        flat = values.detach().to(
            device=self._reward_mean.device,
            dtype=torch.float32,
        ).view(-1)
        if flat.numel() == 0:
            return
        batch_count = torch.as_tensor(float(flat.numel()), device=flat.device)
        batch_mean = flat.mean()
        batch_m2 = ((flat - batch_mean) ** 2).sum()
        if float(self._reward_count.item()) == 0.0:
            self._reward_mean.copy_(batch_mean)
            self._reward_m2.copy_(batch_m2)
            self._reward_count.copy_(batch_count)
            return
        delta = batch_mean - self._reward_mean
        total_count = self._reward_count + batch_count
        self._reward_mean.add_(delta * batch_count / total_count)
        self._reward_m2.add_(
            batch_m2
            + delta.pow(2) * self._reward_count * batch_count / total_count
        )
        self._reward_count.copy_(total_count)


class _RNDNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        embedding_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        channels = int(input_shape[0])
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        feature_size = _feature_size(self.features, input_shape)
        self.head = nn.Sequential(
            nn.Linear(feature_size, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(embedding_size)),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(observation))


def exploration_summary(config: ExplorationConfig | Any) -> dict[str, Any]:
    """Return JSON/CSV-friendly exploration metadata."""
    exploration = getattr(config, "exploration", config)
    return {
        "exploration_enabled": bool(exploration.enabled),
        "exploration_method": str(exploration.method),
        "exploration_observation_source": str(exploration.observation_source),
        "intrinsic_reward_scale": float(exploration.intrinsic_reward_scale),
        "intrinsic_reward_clip": exploration.intrinsic_reward_clip,
        "normalize_intrinsic_rewards": bool(exploration.normalize_intrinsic_rewards),
        "normalize_exploration_observations": bool(
            exploration.normalize_observations
        ),
        "rnd_predictor_learning_rate": float(
            exploration.predictor_learning_rate
        ),
        "rnd_warmup_steps": int(exploration.warmup_steps),
        "rnd_warmup_reward_scale": float(exploration.warmup_reward_scale),
    }


def _feature_size(features: nn.Module, input_shape: tuple[int, int, int]) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape, dtype=torch.float32)
        output = features(dummy)
    return int(output.numel())


__all__ = [
    "ExplorationConfig",
    "RND_OBSERVATION_SOURCE",
    "RNDReward",
    "RandomNetworkDistillation",
    "exploration_summary",
]
