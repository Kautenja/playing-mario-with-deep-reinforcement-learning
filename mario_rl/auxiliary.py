"""Auxiliary prediction targets for recurrent Mario policies."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


GAME_FAMILY_CLASSES = ("smb1", "lost_levels", "smb2_usa", "smb3", "unknown")
REGRESSION_TARGETS = (
    "progress_delta",
    "progress_normalized",
    "transformed_reward",
    "reward_total_unclipped",
    "reward_total_clipped",
)
BINARY_TARGETS = ("clear", "death")
CLASSIFICATION_TARGETS = ("game_family",)
SUPPORTED_AUXILIARY_TARGETS = (
    *REGRESSION_TARGETS,
    *BINARY_TARGETS,
    *CLASSIFICATION_TARGETS,
)
DEFAULT_AUXILIARY_TARGETS = (
    "progress_delta",
    "clear",
    "death",
    "transformed_reward",
    "game_family",
)
_GAME_FAMILY_TO_INDEX = {
    family: index for index, family in enumerate(GAME_FAMILY_CLASSES)
}


@dataclass(frozen=True)
class AuxiliaryLossConfig:
    """Optional supervised auxiliary losses for recurrent actor-critic training."""

    enabled: bool = False
    targets: tuple[str, ...] = ()
    weights: dict[str, float] = field(default_factory=dict)
    head_hidden_size: int = 64

    def __post_init__(self) -> None:
        enabled = _bool(self.enabled)
        targets = _str_tuple(self.targets)
        if enabled and not targets:
            targets = DEFAULT_AUXILIARY_TARGETS
        unknown_targets = set(targets) - set(SUPPORTED_AUXILIARY_TARGETS)
        if unknown_targets:
            names = ", ".join(sorted(unknown_targets))
            choices = ", ".join(SUPPORTED_AUXILIARY_TARGETS)
            raise ValueError(
                f"unknown auxiliary target(s): {names}; choose from {choices}"
            )

        weights = {str(name): float(weight) for name, weight in self.weights.items()}
        unknown_weights = set(weights) - set(SUPPORTED_AUXILIARY_TARGETS)
        if unknown_weights:
            names = ", ".join(sorted(unknown_weights))
            choices = ", ".join(SUPPORTED_AUXILIARY_TARGETS)
            raise ValueError(
                f"unknown auxiliary weight target(s): {names}; choose from {choices}"
            )
        negative_weights = {
            name: weight for name, weight in weights.items() if float(weight) < 0.0
        }
        if negative_weights:
            names = ", ".join(sorted(negative_weights))
            raise ValueError(f"auxiliary weights must be non-negative: {names}")

        head_hidden_size = int(self.head_hidden_size)
        if head_hidden_size <= 0:
            raise ValueError("auxiliary head_hidden_size must be > 0")

        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "head_hidden_size", head_hidden_size)


@dataclass(frozen=True)
class AuxiliaryTargetRecord:
    """One environment step's auxiliary target values and availability masks."""

    values: dict[str, float]
    masks: dict[str, bool]


def auxiliary_target_names(config: AuxiliaryLossConfig) -> tuple[str, ...]:
    """Return enabled auxiliary target names, or an empty tuple when disabled."""
    return tuple(config.targets) if config.enabled else ()


def auxiliary_loss_weights(config: AuxiliaryLossConfig) -> dict[str, float]:
    """Return per-target weights for enabled targets."""
    return {target: float(config.weights.get(target, 1.0)) for target in config.targets}


def auxiliary_output_sizes(
    config_or_targets: AuxiliaryLossConfig | Sequence[str],
) -> dict[str, int]:
    """Return model output widths for auxiliary target heads."""
    if isinstance(config_or_targets, AuxiliaryLossConfig):
        targets = auxiliary_target_names(config_or_targets)
    else:
        targets = tuple(str(target) for target in config_or_targets)
    return {
        target: len(GAME_FAMILY_CLASSES) if target in CLASSIFICATION_TARGETS else 1
        for target in targets
    }


def auxiliary_target_kind(target: str) -> str:
    """Return the loss family for ``target``."""
    if target in REGRESSION_TARGETS:
        return "regression"
    if target in BINARY_TARGETS:
        return "binary"
    if target in CLASSIFICATION_TARGETS:
        return "classification"
    raise ValueError(f"unknown auxiliary target {target!r}")


def extract_auxiliary_targets(
    info: Mapping[str, Any] | None,
    *,
    transformed_reward: float | None = None,
    targets: Sequence[str] = SUPPORTED_AUXILIARY_TARGETS,
) -> AuxiliaryTargetRecord:
    """Extract configured target values with explicit missing-value masks."""
    info_map = info if isinstance(info, Mapping) else {}
    components = _float_mapping(info_map.get("reward_components"))
    progress = _first_optional_float(
        info_map.get("progress"),
        info_map.get("x_pos"),
        info_map.get("position_progress"),
    )
    progress_max = _first_optional_float(
        info_map.get("progress_max"),
        info_map.get("x_pos_max"),
        info_map.get("position_progress_max"),
    )

    values: dict[str, float] = {}
    masks: dict[str, bool] = {}
    for target in targets:
        if target == "progress_delta":
            value = components.get("progress") if components is not None else None
            _set_optional(values, masks, target, value)
        elif target == "progress_normalized":
            if progress is not None and progress_max is not None and progress_max > 0.0:
                _set_optional(values, masks, target, progress / progress_max)
            else:
                _set_missing(values, masks, target)
        elif target == "clear":
            _set_optional_bool(values, masks, target, info_map, "clear")
        elif target == "death":
            _set_optional_bool(values, masks, target, info_map, "death")
        elif target == "transformed_reward":
            _set_optional(values, masks, target, transformed_reward)
        elif target == "reward_total_unclipped":
            _set_optional(values, masks, target, info_map.get("reward_total_unclipped"))
        elif target == "reward_total_clipped":
            _set_optional(values, masks, target, info_map.get("reward_total_clipped"))
        elif target == "game_family":
            _set_game_family(values, masks, target, info_map)
        else:
            raise ValueError(f"unknown auxiliary target {target!r}")
    return AuxiliaryTargetRecord(values=values, masks=masks)


def _set_optional(
    values: dict[str, float],
    masks: dict[str, bool],
    target: str,
    value: Any,
) -> None:
    parsed = _optional_float(value)
    if parsed is None:
        _set_missing(values, masks, target)
        return
    values[target] = float(parsed)
    masks[target] = True


def _set_optional_bool(
    values: dict[str, float],
    masks: dict[str, bool],
    target: str,
    info_map: Mapping[str, Any],
    key: str,
) -> None:
    if key not in info_map:
        _set_missing(values, masks, target)
        return
    values[target] = 1.0 if bool(info_map[key]) else 0.0
    masks[target] = True


def _set_game_family(
    values: dict[str, float],
    masks: dict[str, bool],
    target: str,
    info_map: Mapping[str, Any],
) -> None:
    family = info_map.get("game_family")
    if family is None:
        _set_missing(values, masks, target)
        return
    index = _GAME_FAMILY_TO_INDEX.get(str(family), _GAME_FAMILY_TO_INDEX["unknown"])
    values[target] = float(index)
    masks[target] = True


def _set_missing(
    values: dict[str, float],
    masks: dict[str, bool],
    target: str,
) -> None:
    values[target] = 0.0
    masks[target] = False


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"expected boolean value, got {value!r}")
    return bool(value)


def _str_tuple(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _first_optional_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _float_mapping(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("reward_components must be a mapping")
    return {str(name): float(component) for name, component in value.items()}


__all__ = [
    "AuxiliaryLossConfig",
    "AuxiliaryTargetRecord",
    "BINARY_TARGETS",
    "CLASSIFICATION_TARGETS",
    "DEFAULT_AUXILIARY_TARGETS",
    "GAME_FAMILY_CLASSES",
    "REGRESSION_TARGETS",
    "SUPPORTED_AUXILIARY_TARGETS",
    "auxiliary_loss_weights",
    "auxiliary_output_sizes",
    "auxiliary_target_kind",
    "auxiliary_target_names",
    "extract_auxiliary_targets",
]
