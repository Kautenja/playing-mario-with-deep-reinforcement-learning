"""PyTorch model definitions and factories."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from mario_rl.auxiliary import auxiliary_output_sizes
from mario_rl.config import AUTO_NUM_ACTIONS, resolve_model_num_actions


DEFAULT_INPUT_SHAPE = (4, 84, 84)


@dataclass(frozen=True)
class ActorCriticOutput:
    """Policy/value/recurrent outputs for one actor-critic forward pass."""

    policy_logits: torch.Tensor
    value: torch.Tensor
    hidden_state: torch.Tensor
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)


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
        task_feature_size: int = 0,
        normalize_input: bool = True,
        input_scale: float = 255.0,
    ) -> None:
        super().__init__()
        input_shape = _canonical_input_shape(input_shape, input_channels)
        self.input_shape = input_shape
        self.num_actions = int(num_actions)
        self.task_feature_size = int(task_feature_size)
        self.normalize_input = bool(normalize_input)
        self.input_scale = float(input_scale)
        self.features = _nature_cnn_features(input_shape[0])
        feature_size = _feature_size(self.features, input_shape) + self.task_feature_size
        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), self.num_actions),
        )

    def forward(
        self,
        observation: torch.Tensor,
        task_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return Q-values for channel-first observations."""
        x = self._prepare_observation(observation)
        features = _append_task_features(
            self.features(x),
            task_features=task_features,
            task_feature_size=self.task_feature_size,
        )
        return self.q_head(features)

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
        task_feature_size: int = 0,
        normalize_input: bool = True,
        input_scale: float = 255.0,
    ) -> None:
        super().__init__()
        input_shape = _canonical_input_shape(input_shape, input_channels)
        self.input_shape = input_shape
        self.num_actions = int(num_actions)
        self.task_feature_size = int(task_feature_size)
        self.normalize_input = bool(normalize_input)
        self.input_scale = float(input_scale)
        self.features = _nature_cnn_features(input_shape[0])
        feature_size = _feature_size(self.features, input_shape) + self.task_feature_size
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

    def forward(
        self,
        observation: torch.Tensor,
        task_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return dueling Q-values for channel-first observations."""
        x = self._prepare_observation(observation)
        features = _append_task_features(
            self.features(x),
            task_features=task_features,
            task_feature_size=self.task_feature_size,
        )
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


class RecurrentActorCritic(nn.Module):
    """Convolutional recurrent actor-critic for channel-first Mario frames."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_actions: int = 7,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        hidden_size: int = 512,
        recurrent_hidden_size: int = 256,
        task_feature_size: int = 0,
        task_embedding_size: int = 32,
        auxiliary_outputs: Mapping[str, int] | None = None,
        auxiliary_hidden_size: int = 64,
        normalize_input: bool = True,
        input_scale: float = 255.0,
    ) -> None:
        super().__init__()
        input_shape = _canonical_input_shape(input_shape, input_channels)
        self.input_shape = input_shape
        self.num_actions = int(num_actions)
        self.task_feature_size = int(task_feature_size)
        self.task_embedding_size = int(task_embedding_size)
        self.recurrent_hidden_size = int(recurrent_hidden_size)
        self.auxiliary_outputs = {
            str(name): int(width) for name, width in dict(auxiliary_outputs or {}).items()
        }
        self._auxiliary_head_keys = {
            name: f"auxiliary__{name}" for name in self.auxiliary_outputs
        }
        self.auxiliary_hidden_size = int(auxiliary_hidden_size)
        self.normalize_input = bool(normalize_input)
        self.input_scale = float(input_scale)

        self.features = _nature_cnn_features(input_shape[0])
        visual_size = _feature_size(self.features, input_shape)
        if self.task_feature_size > 0:
            self.task_embedding = nn.Sequential(
                nn.Linear(self.task_feature_size, self.task_embedding_size),
                nn.Tanh(),
            )
            recurrent_input_size = visual_size + self.task_embedding_size
        else:
            self.task_embedding = None
            recurrent_input_size = visual_size

        self.visual_projection = nn.Sequential(
            nn.Linear(recurrent_input_size, int(hidden_size)),
            nn.ReLU(),
        )
        self.memory = nn.GRU(int(hidden_size), self.recurrent_hidden_size)
        self.policy_head = nn.Linear(self.recurrent_hidden_size, self.num_actions)
        self.value_head = nn.Linear(self.recurrent_hidden_size, 1)
        self.auxiliary_heads = nn.ModuleDict(
            {
                self._auxiliary_head_keys[name]: nn.Sequential(
                    nn.Linear(self.recurrent_hidden_size, self.auxiliary_hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.auxiliary_hidden_size, output_size),
                )
                for name, output_size in self.auxiliary_outputs.items()
            }
        )

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return a zeroed recurrent hidden state for ``batch_size`` actors."""
        return torch.zeros(
            1,
            int(batch_size),
            self.recurrent_hidden_size,
            dtype=torch.float32,
            device=device,
        )

    def forward(
        self,
        observation: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        task_features: torch.Tensor | None = None,
    ) -> ActorCriticOutput:
        """Return policy logits, values, and next recurrent state."""
        sequence, single_step = self._prepare_observation_sequence(observation)
        steps, batch_size = sequence.shape[:2]
        flat_observation = sequence.reshape(steps * batch_size, *sequence.shape[2:])
        x = self._prepare_observation(flat_observation)
        visual = torch.flatten(self.features(x), start_dim=1)
        task_embedding = self._task_embedding(
            task_features,
            steps=steps,
            batch_size=batch_size,
            device=visual.device,
        )
        if task_embedding is not None:
            visual = torch.cat((visual, task_embedding), dim=1)
        projected = self.visual_projection(visual).reshape(steps, batch_size, -1)
        if hidden_state is None:
            hidden_state = self.initial_state(batch_size, device=projected.device)
        else:
            hidden_state = hidden_state.to(device=projected.device, dtype=torch.float32)
        memory_output, next_hidden = self.memory(projected, hidden_state)
        logits = self.policy_head(memory_output)
        value = self.value_head(memory_output).squeeze(-1)
        auxiliary = self._auxiliary_outputs(memory_output)
        if single_step:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
            auxiliary = {
                name: prediction.squeeze(0)
                for name, prediction in auxiliary.items()
            }
        return ActorCriticOutput(
            policy_logits=logits,
            value=value,
            hidden_state=next_hidden,
            auxiliary=auxiliary,
        )

    def reset_recurrent_state(
        self,
        hidden_state: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Zero recurrent state entries whose actors just ended an episode."""
        return reset_recurrent_state(hidden_state, dones)

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            return normalize_observation(observation, self.input_scale)
        return observation.to(dtype=torch.float32)

    def _prepare_observation_sequence(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        tensor = torch.as_tensor(observation)
        if tensor.ndim == 4:
            return tensor.unsqueeze(0), True
        if tensor.ndim == 5:
            return tensor, False
        raise ValueError(
            "observation must have shape (batch, channels, height, width) or "
            f"(steps, batch, channels, height, width), got {tuple(tensor.shape)}"
        )

    def _task_embedding(
        self,
        task_features: torch.Tensor | None,
        *,
        steps: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.task_embedding is None:
            return None
        task_tensor = _prepare_sequence_task_features(
            task_features,
            steps=steps,
            batch_size=batch_size,
            feature_size=self.task_feature_size,
            device=device,
        )
        return self.task_embedding(task_tensor.reshape(steps * batch_size, -1))

    def _auxiliary_outputs(
        self,
        memory_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.auxiliary_heads:
            return {}
        steps, batch_size = memory_output.shape[:2]
        flat_memory = memory_output.reshape(steps * batch_size, -1)
        outputs = {}
        for name, output_size in self.auxiliary_outputs.items():
            head = self.auxiliary_heads[self._auxiliary_head_keys[name]]
            prediction = head(flat_memory)
            output_size = int(output_size)
            prediction = prediction.reshape(steps, batch_size, output_size)
            if output_size == 1:
                prediction = prediction.squeeze(-1)
            outputs[name] = prediction
        return outputs


def reset_recurrent_state(
    hidden_state: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    """Return ``hidden_state`` with actor columns zeroed where ``dones`` is true."""
    if hidden_state.ndim != 3:
        raise ValueError(
            f"hidden_state must have shape (layers, batch, hidden), got {tuple(hidden_state.shape)}"
        )
    mask = torch.as_tensor(dones, dtype=torch.bool, device=hidden_state.device).view(-1)
    if mask.shape[0] != hidden_state.shape[1]:
        raise ValueError("dones batch dimension must match hidden_state")
    keep = (~mask).to(dtype=hidden_state.dtype).view(1, -1, 1)
    return hidden_state * keep


def build_model(
    config: Any = None,
    *,
    architecture: str | None = None,
    input_shape: Sequence[int] | None = None,
    input_channels: int | None = None,
    num_actions: int | None = None,
    hidden_size: int | None = None,
    task_feature_size: int | None = None,
) -> nn.Module:
    """Build a model module from a typed config or explicit keyword values."""
    model_config = getattr(config, "model", config)
    replay_config = getattr(config, "replay", None)
    architecture = architecture or getattr(model_config, "architecture", "dqn")
    input_shape = input_shape or getattr(replay_config, "state_shape", DEFAULT_INPUT_SHAPE)
    input_channels = int(input_channels or getattr(model_config, "input_channels", input_shape[0]))
    input_shape = _canonical_input_shape(input_shape, input_channels)
    if num_actions is None:
        raw_num_actions = getattr(model_config, "num_actions", 7)
        if raw_num_actions == AUTO_NUM_ACTIONS:
            num_actions = resolve_model_num_actions(config)
        else:
            num_actions = int(raw_num_actions)
    else:
        num_actions = int(num_actions)
    hidden_size = int(hidden_size or getattr(model_config, "hidden_size", 512))
    task_conditioning = bool(getattr(model_config, "task_conditioning", False))
    resolved_task_feature_size = _resolve_task_feature_size(
        model_config,
        task_feature_size=task_feature_size,
        task_conditioning=task_conditioning,
    )
    recurrent_hidden_size = int(getattr(model_config, "recurrent_hidden_size", 256))
    task_embedding_size = int(getattr(model_config, "task_embedding_size", 32))
    auxiliary_config = getattr(config, "auxiliary", None)
    auxiliary_outputs = auxiliary_output_sizes(auxiliary_config) if auxiliary_config else {}
    auxiliary_hidden_size = int(getattr(auxiliary_config, "head_hidden_size", 64))
    kwargs = {
        "input_channels": input_channels,
        "num_actions": num_actions,
        "input_shape": input_shape,
        "hidden_size": hidden_size,
        "task_feature_size": resolved_task_feature_size,
    }
    normalized = str(architecture).lower().replace("-", "_")
    if normalized in {"dqn", "deep_q", "deep_q_network"}:
        return DQN(**kwargs)
    if normalized in {"dueling", "dueling_dqn", "dueling_deep_q"}:
        return DuelingDQN(**kwargs)
    if normalized in {
        "actor_critic",
        "recurrent_actor_critic",
        "ppo",
        "ppo_actor_critic",
    }:
        return RecurrentActorCritic(
            **kwargs,
            recurrent_hidden_size=recurrent_hidden_size,
            task_embedding_size=task_embedding_size,
            auxiliary_outputs=auxiliary_outputs,
            auxiliary_hidden_size=auxiliary_hidden_size,
        )
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


def _append_task_features(
    visual_features: torch.Tensor,
    *,
    task_features: torch.Tensor | None,
    task_feature_size: int,
) -> torch.Tensor:
    if task_feature_size <= 0:
        return visual_features
    flat_visual = torch.flatten(visual_features, start_dim=1)
    task_tensor = _prepare_task_features(
        task_features,
        batch_size=flat_visual.shape[0],
        feature_size=task_feature_size,
        device=flat_visual.device,
    )
    return torch.cat((flat_visual, task_tensor), dim=1)


def _prepare_task_features(
    task_features: torch.Tensor | None,
    *,
    batch_size: int,
    feature_size: int,
    device: torch.device,
) -> torch.Tensor:
    if task_features is None:
        return torch.zeros(batch_size, feature_size, dtype=torch.float32, device=device)
    tensor = torch.as_tensor(task_features, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            "task_features must have shape (features,) or "
            f"(batch, features), got {tuple(tensor.shape)}"
        )
    if tensor.shape[1] != feature_size:
        raise ValueError(
            f"expected task feature width {feature_size}, got {tensor.shape[1]}"
        )
    if tensor.shape[0] == 1 and batch_size != 1:
        tensor = tensor.expand(batch_size, -1)
    if tensor.shape[0] != batch_size:
        raise ValueError(
            "task_features batch dimension "
            f"{tensor.shape[0]} does not match observations {batch_size}"
        )
    return tensor


def _prepare_sequence_task_features(
    task_features: torch.Tensor | None,
    *,
    steps: int,
    batch_size: int,
    feature_size: int,
    device: torch.device,
) -> torch.Tensor:
    if task_features is None:
        return torch.zeros(
            steps,
            batch_size,
            feature_size,
            dtype=torch.float32,
            device=device,
        )
    tensor = torch.as_tensor(task_features, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.view(1, 1, feature_size).expand(steps, batch_size, -1)
    elif tensor.ndim == 2:
        if tensor.shape == (batch_size, feature_size):
            tensor = tensor.unsqueeze(0).expand(steps, -1, -1)
        elif tensor.shape == (steps * batch_size, feature_size):
            tensor = tensor.view(steps, batch_size, feature_size)
        elif tensor.shape == (1, feature_size):
            tensor = tensor.view(1, 1, feature_size).expand(steps, batch_size, -1)
        else:
            raise ValueError(
                "task_features must have shape (features,), (batch, features), "
                "(steps * batch, features), or (steps, batch, features)"
            )
    elif tensor.ndim == 3:
        if tensor.shape != (steps, batch_size, feature_size):
            raise ValueError(
                f"expected task feature shape {(steps, batch_size, feature_size)}, "
                f"got {tuple(tensor.shape)}"
            )
    else:
        raise ValueError(
            "task_features must have shape (features,), (batch, features), "
            "(steps * batch, features), or (steps, batch, features)"
        )
    return tensor


def _resolve_task_feature_size(
    model_config: Any,
    *,
    task_feature_size: int | None,
    task_conditioning: bool,
) -> int:
    if not task_conditioning:
        return 0
    configured = task_feature_size
    if configured is None:
        configured = getattr(model_config, "task_feature_size", 0)
    configured = int(configured or 0)
    if configured > 0:
        return configured
    from mario_rl.envs import task_feature_size as default_task_feature_size

    return int(default_task_feature_size())


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
