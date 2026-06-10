"""Reward transform policies for Mario training."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


REWARD_TRANSFORM_MODES = (
    "env",
    "sign",
    "unclipped",
    "clipped",
    "component_weights",
)
MISSING_TOTAL_POLICIES = ("error", "env")
MISSING_COMPONENT_POLICIES = ("zero", "error")


@dataclass(frozen=True)
class RewardTransformConfig:
    """Training-reward transform and diagnostic extraction settings."""

    mode: str = "env"
    missing_total_policy: str = "error"
    component_weights: dict[str, float] = field(default_factory=dict)
    missing_component_policy: str = "zero"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip()
        if mode not in REWARD_TRANSFORM_MODES:
            choices = ", ".join(REWARD_TRANSFORM_MODES)
            raise ValueError(f"unknown reward transform mode {mode!r}; choose {choices}")
        missing_total_policy = str(self.missing_total_policy).strip()
        if missing_total_policy not in MISSING_TOTAL_POLICIES:
            choices = ", ".join(MISSING_TOTAL_POLICIES)
            raise ValueError(
                "unknown missing total policy "
                f"{missing_total_policy!r}; choose {choices}"
            )
        missing_component_policy = str(self.missing_component_policy).strip()
        if missing_component_policy not in MISSING_COMPONENT_POLICIES:
            choices = ", ".join(MISSING_COMPONENT_POLICIES)
            raise ValueError(
                "unknown missing component policy "
                f"{missing_component_policy!r}; choose {choices}"
            )
        weights = {
            str(name): float(weight)
            for name, weight in dict(self.component_weights).items()
        }
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "missing_total_policy", missing_total_policy)
        object.__setattr__(self, "component_weights", weights)
        object.__setattr__(
            self,
            "missing_component_policy",
            missing_component_policy,
        )


@dataclass(frozen=True)
class RewardTransformResult:
    """One transformed reward plus optional debug reward diagnostics."""

    training_reward: float
    env_reward: float
    raw_reward: float
    unclipped_reward: float | None
    clipped_reward: float | None
    reward_components: dict[str, float] | None


class RewardTransformer:
    """Apply configured reward transforms to Gymnasium step outputs."""

    def __init__(self, config: RewardTransformConfig) -> None:
        self.config = config

    def transform(
        self,
        reward: float,
        info: Mapping[str, Any] | None,
    ) -> RewardTransformResult:
        """Return the reward that should enter replay and training metrics."""
        env_reward = float(reward)
        info_map = info if isinstance(info, Mapping) else {}
        raw_reward = _optional_float(info_map.get("raw_reward"))
        if raw_reward is None:
            raw_reward = env_reward
        unclipped_reward = _optional_float(info_map.get("reward_total_unclipped"))
        clipped_reward = _optional_float(info_map.get("reward_total_clipped"))
        components = _reward_components(info_map.get("reward_components"))

        mode = self.config.mode
        if mode == "env":
            training_reward = env_reward
        elif mode == "sign":
            training_reward = float(np.sign(env_reward))
        elif mode == "unclipped":
            training_reward = self._total_or_fallback(
                "reward_total_unclipped",
                unclipped_reward,
                env_reward,
            )
        elif mode == "clipped":
            training_reward = self._total_or_fallback(
                "reward_total_clipped",
                clipped_reward,
                env_reward,
            )
        elif mode == "component_weights":
            training_reward = self._weighted_components(components)
        else:  # pragma: no cover - RewardTransformConfig validates this.
            raise AssertionError(f"unhandled reward transform mode {mode!r}")

        return RewardTransformResult(
            training_reward=float(training_reward),
            env_reward=env_reward,
            raw_reward=float(raw_reward),
            unclipped_reward=unclipped_reward,
            clipped_reward=clipped_reward,
            reward_components=components,
        )

    def _total_or_fallback(
        self,
        field_name: str,
        value: float | None,
        env_reward: float,
    ) -> float:
        if value is not None:
            return float(value)
        if self.config.missing_total_policy == "env":
            return float(env_reward)
        raise KeyError(f"reward info is missing {field_name!r}")

    def _weighted_components(self, components: dict[str, float] | None) -> float:
        if not self.config.component_weights:
            return 0.0
        if components is None:
            if self.config.missing_component_policy == "zero":
                return 0.0
            raise KeyError("reward info is missing 'reward_components'")

        total = 0.0
        for name, weight in self.config.component_weights.items():
            if name in components:
                total += float(weight) * float(components[name])
            elif self.config.missing_component_policy == "error":
                raise KeyError(f"reward component {name!r} is missing")
        return float(total)


def reward_transform_summary(config: RewardTransformConfig) -> dict[str, Any]:
    """Return JSON/CSV-friendly reward-transform metadata."""
    return {
        "reward_transform_mode": config.mode,
        "reward_missing_total_policy": config.missing_total_policy,
        "reward_missing_component_policy": config.missing_component_policy,
        "reward_component_weights": dict(config.component_weights),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _reward_components(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("reward_components must be a mapping")
    return {str(name): float(component) for name, component in value.items()}


__all__ = [
    "MISSING_COMPONENT_POLICIES",
    "MISSING_TOTAL_POLICIES",
    "REWARD_TRANSFORM_MODES",
    "RewardTransformConfig",
    "RewardTransformResult",
    "RewardTransformer",
    "reward_transform_summary",
]
