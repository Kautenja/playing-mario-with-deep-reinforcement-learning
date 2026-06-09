"""PyTorch DQN network definitions and factories."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


DEFAULT_INPUT_SHAPE = (4, 84, 84)


def normalize_observation(observation: torch.Tensor, scale: float = 255.0) -> torch.Tensor:
    """Return observations as ``float32`` values scaled into the training range."""
    return observation.to(dtype=torch.float32) / float(scale)


class DQN(nn.Module):
    """DeepMind-style convolutional DQN for channel-first frame stacks."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_actions: int = 7,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        hidden_size: int = 512,
        normalize_input: bool = True,
        input_scale: float = 255.0,
    ) -> None:
        super().__init__()
        input_shape = _canonical_input_shape(input_shape, input_channels)
        self.input_shape = input_shape
        self.num_actions = int(num_actions)
        self.normalize_input = bool(normalize_input)
        self.input_scale = float(input_scale)
        self.features = _nature_cnn_features(input_shape[0])
        feature_size = _feature_size(self.features, input_shape)
        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), self.num_actions),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return Q-values for channel-first observations."""
        x = self._prepare_observation(observation)
        return self.q_head(self.features(x))

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return normalize_observation(observation, self.input_scale)
        return observation.to(dtype=torch.float32)


class DuelingDQN(nn.Module):
    """Dueling DQN with separate value and advantage streams."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_actions: int = 7,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        hidden_size: int = 512,
        normalize_input: bool = True,
        input_scale: float = 255.0,
    ) -> None:
        super().__init__()
        input_shape = _canonical_input_shape(input_shape, input_channels)
        self.input_shape = input_shape
        self.num_actions = int(num_actions)
        self.normalize_input = bool(normalize_input)
        self.input_scale = float(input_scale)
        self.features = _nature_cnn_features(input_shape[0])
        feature_size = _feature_size(self.features, input_shape)
        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), self.num_actions),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return dueling Q-values for channel-first observations."""
        x = self._prepare_observation(observation)
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return self.combine_value_advantage(value, advantage)

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return normalize_observation(observation, self.input_scale)
        return observation.to(dtype=torch.float32)

    @staticmethod
    def combine_value_advantage(value: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        """Combine value and advantage streams using mean-centered advantages."""
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_model(
    config: Any = None,
    *,
    architecture: str | None = None,
    input_shape: Sequence[int] | None = None,
    input_channels: int | None = None,
    num_actions: int | None = None,
    hidden_size: int | None = None,
) -> nn.Module:
    """Build a DQN module from a typed config or explicit keyword values."""
    model_config = getattr(config, "model", config)
    replay_config = getattr(config, "replay", None)
    architecture = architecture or getattr(model_config, "architecture", "dqn")
    input_shape = input_shape or getattr(replay_config, "state_shape", DEFAULT_INPUT_SHAPE)
    input_channels = int(input_channels or getattr(model_config, "input_channels", input_shape[0]))
    input_shape = _canonical_input_shape(input_shape, input_channels)
    num_actions = int(num_actions or getattr(model_config, "num_actions", 7))
    hidden_size = int(hidden_size or getattr(model_config, "hidden_size", 512))
    kwargs = {
        "input_channels": input_channels,
        "num_actions": num_actions,
        "input_shape": input_shape,
        "hidden_size": hidden_size,
    }
    normalized = str(architecture).lower().replace("-", "_")
    if normalized in {"dqn", "deep_q", "deep_q_network"}:
        return DQN(**kwargs)
    if normalized in {"dueling", "dueling_dqn", "dueling_deep_q"}:
        return DuelingDQN(**kwargs)
    raise ValueError(f"unsupported model architecture: {architecture!r}")


def make_optimizer(model: nn.Module, config: Any = None) -> torch.optim.Optimizer:
    """Create the configured PyTorch optimizer for a model."""
    model_config = getattr(config, "model", config)
    name = str(getattr(model_config, "optimizer", "adam")).lower()
    learning_rate = float(getattr(model_config, "learning_rate", 0.00025))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=0.01)
    raise ValueError(f"unsupported optimizer: {name!r}")


def _nature_cnn_features(input_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(int(input_channels), 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
    )


def _feature_size(features: nn.Module, input_shape: tuple[int, int, int]) -> int:
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape, dtype=torch.float32)
        output = features(dummy)
    return int(output.numel())


def _canonical_input_shape(
    input_shape: Sequence[int],
    input_channels: int | None = None,
) -> tuple[int, int, int]:
    shape = tuple(int(dimension) for dimension in input_shape)
    if len(shape) != 3:
        raise ValueError(f"expected channel-first input shape of length 3, got {shape!r}")
    if input_channels is not None and shape[0] != int(input_channels):
        raise ValueError(
            f"input_shape channels {shape[0]} do not match input_channels {input_channels}"
        )
    return shape
