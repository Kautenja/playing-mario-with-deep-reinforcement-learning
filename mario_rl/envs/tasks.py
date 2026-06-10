"""Task metadata and conditioning helpers for the Mario environment surface."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Any

import gym_super_mario_bros
import numpy as np
import torch


MarioTask = gym_super_mario_bros.MarioTask
smb3_stage_matrix = gym_super_mario_bros.smb3_stage_matrix
UNKNOWN_TASK_VALUE = "<unknown>"


@dataclass(frozen=True)
class TaskFeatures:
    """Encoded task metadata plus the dense feature vector used by models."""

    env_id: str | None
    task_id: str
    game_family: str
    rom_mode: str
    world: int | None
    stage: int | None
    single_stage: bool
    validated: bool
    vector: np.ndarray

    def to_tensor(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Return this feature vector as a float32 torch tensor."""
        return torch.as_tensor(self.vector, dtype=torch.float32, device=device)


class TaskFeatureEncoder:
    """
    Deterministically map :class:`MarioTask` metadata into model features.

    Categorical values use stable one-hot vocabularies built from the
    registered task catalog. Unknown/custom environment IDs are encoded through
    the first vocabulary slot and zero-valued numeric fields.
    """

    numeric_feature_names = (
        "world_norm",
        "stage_norm",
        "has_world",
        "has_stage",
        "single_stage",
        "validated",
    )

    def __init__(self, tasks: Iterable[MarioTask] | None = None) -> None:
        catalog = tuple(tasks) if tasks is not None else available_tasks(include_aliases=True)
        if not catalog:
            raise ValueError("TaskFeatureEncoder requires at least one task")
        self.game_families = _vocabulary(task.game_family for task in catalog)
        self.task_ids = _vocabulary(task.task_id for task in catalog)
        self.rom_modes = _vocabulary(task.rom_mode for task in catalog)
        self.max_world = max(
            (int(task.world) for task in catalog if task.world is not None),
            default=1,
        )
        self.max_stage = max(
            (int(task.stage) for task in catalog if task.stage is not None),
            default=1,
        )
        self.feature_names = (
            *(f"game_family={name}" for name in self.game_families),
            *(f"task_id={name}" for name in self.task_ids),
            *(f"rom_mode={name}" for name in self.rom_modes),
            *self.numeric_feature_names,
        )

    @property
    def feature_size(self) -> int:
        """Return the width of encoded task feature vectors."""
        return len(self.feature_names)

    def encode_env_id(self, env_id: str) -> TaskFeatures:
        """Encode a registered environment ID or the explicit unknown fallback."""
        try:
            task = task_for_env_id(env_id)
        except KeyError:
            return self.encode_custom(env_id=env_id)
        return self.encode_task(task)

    def encode_task(self, task: MarioTask) -> TaskFeatures:
        """Encode registered Mario task metadata without constructing an environment."""
        return self.encode_custom(
            env_id=task.env_id,
            task_id=task.task_id,
            game_family=task.game_family,
            rom_mode=task.rom_mode,
            world=task.world,
            stage=task.stage,
            single_stage=task.single_stage,
            validated=task.validated,
        )

    def encode_custom(
        self,
        *,
        env_id: str | None = None,
        task_id: str | None = None,
        game_family: str | None = None,
        rom_mode: str | None = None,
        world: int | None = None,
        stage: int | None = None,
        single_stage: bool = False,
        validated: bool = False,
    ) -> TaskFeatures:
        """Encode user-provided task metadata with explicit unknown handling."""
        normalized_family = _known_or_unknown(game_family, self.game_families)
        normalized_task_id = _known_or_unknown(task_id, self.task_ids)
        normalized_rom_mode = _known_or_unknown(rom_mode, self.rom_modes)
        vector = np.concatenate(
            (
                _one_hot(normalized_family, self.game_families),
                _one_hot(normalized_task_id, self.task_ids),
                _one_hot(normalized_rom_mode, self.rom_modes),
                np.asarray(
                    (
                        _normalized_int(world, self.max_world),
                        _normalized_int(stage, self.max_stage),
                        1.0 if world is not None else 0.0,
                        1.0 if stage is not None else 0.0,
                        1.0 if single_stage else 0.0,
                        1.0 if validated else 0.0,
                    ),
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32, copy=False)
        return TaskFeatures(
            env_id=env_id,
            task_id=normalized_task_id,
            game_family=normalized_family,
            rom_mode=normalized_rom_mode,
            world=int(world) if world is not None else None,
            stage=int(stage) if stage is not None else None,
            single_stage=bool(single_stage),
            validated=bool(validated),
            vector=vector,
        )

    def encode_mapping(self, data: Mapping[str, Any]) -> TaskFeatures:
        """Encode task-like dictionaries such as Gymnasium info payloads."""
        if "env_id" in data:
            task = task_for_env_id_or_none(str(data["env_id"]))
            if task is not None:
                return self.encode_task(task)
        return self.encode_custom(
            env_id=_optional_str(data.get("env_id")),
            task_id=_optional_str(data.get("task_id")),
            game_family=_optional_str(data.get("game_family")),
            rom_mode=_optional_str(data.get("rom_mode")),
            world=data.get("world"),
            stage=data.get("stage"),
            single_stage=bool(data.get("single_stage", False)),
            validated=bool(data.get("validated", False)),
        )


def available_tasks(**filters: Any) -> tuple[MarioTask, ...]:
    """Return registered Mario task metadata matching gym-super-mario-bros filters."""
    return tuple(gym_super_mario_bros.all_tasks(**filters))


def available_env_ids(**filters: Any) -> tuple[str, ...]:
    """Return registered Mario environment IDs matching gym-super-mario-bros filters."""
    return tuple(gym_super_mario_bros.task_ids(**filters))


def task_for_env_id(env_id: str) -> MarioTask:
    """Return task metadata for a registered Mario environment ID."""
    return gym_super_mario_bros.task_for_env_id(env_id)


def task_for_env_id_or_none(env_id: str) -> MarioTask | None:
    """Return task metadata for a registered ID, or ``None`` for custom IDs."""
    try:
        return task_for_env_id(env_id)
    except KeyError:
        return None


def choose_stage_env_id(
    *,
    seed: int | None = None,
    game_family: str = "smb1",
    include_aliases: bool = False,
    split: str = "train",
    validated: bool = True,
) -> str:
    """
    Select a deterministic single-stage environment ID from 9.x task metadata.

    gym-super-mario-bros 9.0.0 removed the old ``SuperMarioBrosRandomStages-*``
    registration family. Use this helper when a config needs seeded stage
    selection without relying on removed environment IDs.
    """
    candidates = available_env_ids(
        include_aliases=include_aliases,
        game_family=game_family,
        single_stage=True,
        split=split,
        validated=validated,
    )
    if not candidates:
        raise ValueError("no Mario stage environments match the requested filters")
    return random.Random(seed).choice(candidates)


def task_feature_size() -> int:
    """Return the default encoded task feature width."""
    return TaskFeatureEncoder().feature_size


def encode_task_features(
    env_id: str,
    *,
    encoder: TaskFeatureEncoder | None = None,
) -> TaskFeatures:
    """Encode an environment ID using the default task feature contract."""
    active_encoder = encoder or TaskFeatureEncoder()
    return active_encoder.encode_env_id(env_id)


def _vocabulary(values: Iterable[str | None]) -> tuple[str, ...]:
    return (UNKNOWN_TASK_VALUE, *sorted({str(value) for value in values if value is not None}))


def _known_or_unknown(value: str | None, vocabulary: Sequence[str]) -> str:
    if value is None:
        return UNKNOWN_TASK_VALUE
    value = str(value)
    if value in vocabulary:
        return value
    return UNKNOWN_TASK_VALUE


def _one_hot(value: str, vocabulary: Sequence[str]) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    vector[vocabulary.index(value)] = 1.0
    return vector


def _normalized_int(value: Any, maximum: int) -> float:
    if value is None:
        return 0.0
    return float(int(value)) / float(max(int(maximum), 1))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "MarioTask",
    "TaskFeatureEncoder",
    "TaskFeatures",
    "UNKNOWN_TASK_VALUE",
    "available_env_ids",
    "available_tasks",
    "choose_stage_env_id",
    "encode_task_features",
    "smb3_stage_matrix",
    "task_feature_size",
    "task_for_env_id",
    "task_for_env_id_or_none",
]
