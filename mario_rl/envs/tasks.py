"""Task metadata and conditioning helpers for the Mario environment surface."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class TaskSuiteConfig:
    """Configuration for metadata-only Mario task-suite resolution and sampling."""

    enabled: bool = False
    game_families: tuple[str, ...] = ()
    single_stage: bool | None = True
    splits: tuple[str, ...] = ("train",)
    exclude_splits: tuple[str, ...] = ()
    include_validated: bool | None = True
    exclude_validated: bool | None = None
    include_aliases: bool = False
    include_env_ids: tuple[str, ...] = ()
    exclude_env_ids: tuple[str, ...] = ()
    include_worlds: tuple[int, ...] = ()
    exclude_worlds: tuple[int, ...] = ()
    include_stages: tuple[int, ...] = ()
    exclude_stages: tuple[int, ...] = ()
    family_weights: dict[str, float] = field(default_factory=dict)
    seed: int | None = None
    switch_interval_episodes: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _bool(self.enabled))
        object.__setattr__(self, "game_families", _str_tuple(self.game_families))
        object.__setattr__(self, "single_stage", _optional_bool(self.single_stage))
        object.__setattr__(self, "splits", _str_tuple(self.splits))
        object.__setattr__(self, "exclude_splits", _str_tuple(self.exclude_splits))
        object.__setattr__(
            self,
            "include_validated",
            _optional_bool(self.include_validated),
        )
        object.__setattr__(
            self,
            "exclude_validated",
            _optional_bool(self.exclude_validated),
        )
        object.__setattr__(self, "include_aliases", _bool(self.include_aliases))
        object.__setattr__(self, "include_env_ids", _str_tuple(self.include_env_ids))
        object.__setattr__(self, "exclude_env_ids", _str_tuple(self.exclude_env_ids))
        object.__setattr__(self, "include_worlds", _int_tuple(self.include_worlds))
        object.__setattr__(self, "exclude_worlds", _int_tuple(self.exclude_worlds))
        object.__setattr__(self, "include_stages", _int_tuple(self.include_stages))
        object.__setattr__(self, "exclude_stages", _int_tuple(self.exclude_stages))
        object.__setattr__(
            self,
            "family_weights",
            _family_weight_mapping(self.family_weights),
        )
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))
        interval = int(self.switch_interval_episodes)
        if interval <= 0:
            raise ValueError("switch_interval_episodes must be > 0")
        object.__setattr__(self, "switch_interval_episodes", interval)


class TaskSuite:
    """Resolve and sample registered Mario tasks without constructing envs."""

    def __init__(
        self,
        config: TaskSuiteConfig | Mapping[str, Any] | None = None,
        *,
        tasks: Iterable[MarioTask] | None = None,
    ) -> None:
        self.config = _coerce_task_suite_config(config)
        self.candidates = self._resolve_candidates(tasks)
        self._tasks_by_family = _group_tasks_by_family(self.candidates)
        self._families = tuple(sorted(self._tasks_by_family))
        self._weights = tuple(
            self.config.family_weights.get(family, 1.0)
            for family in self._families
        )
        if any(weight <= 0.0 for weight in self._weights):
            raise ValueError("family weights must be positive")
        missing_weights = set(self.config.family_weights) - set(self._families)
        if missing_weights:
            names = ", ".join(sorted(missing_weights))
            raise ValueError(f"family_weights reference empty task families: {names}")
        self._sample_index = 0

    @property
    def env_ids(self) -> tuple[str, ...]:
        """Return candidate environment IDs in stable resolver order."""
        return tuple(task.env_id for task in self.candidates)

    @property
    def family_counts(self) -> dict[str, int]:
        """Return candidate counts by game family."""
        return {family: len(tasks) for family, tasks in self._tasks_by_family.items()}

    def task_for_index(self, index: int) -> MarioTask:
        """Return the deterministic sampled task for a zero-based sample index."""
        if index < 0:
            raise ValueError("sample index must be >= 0")
        rng = random.Random(f"{self.config.seed}:{int(index)}")
        family = _weighted_choice(rng, self._families, self._weights)
        return rng.choice(self._tasks_by_family[family])

    def sample(self) -> MarioTask:
        """Return the next deterministic sample and advance the local cursor."""
        task = self.task_for_index(self._sample_index)
        self._sample_index += 1
        return task

    def task_for_episode(self, episode: int) -> MarioTask:
        """Return the active task for an episode number and switch interval."""
        if episode < 0:
            raise ValueError("episode must be >= 0")
        index = int(episode) // int(self.config.switch_interval_episodes)
        return self.task_for_index(index)

    def smb3_catalog(self, *, validated: bool | None = None):
        """Return the full SMB3 stage catalog for reports, not train sampling."""
        return tuple(smb3_stage_matrix(validated=validated))

    def _resolve_candidates(
        self,
        tasks: Iterable[MarioTask] | None,
    ) -> tuple[MarioTask, ...]:
        config = self.config
        _validate_task_suite_filters(config)
        candidates = tuple(tasks) if tasks is not None else available_tasks(
            include_aliases=config.include_aliases
        )
        if not config.include_aliases:
            candidates = tuple(task for task in candidates if task.alias_of is None)
        if config.game_families:
            families = set(config.game_families)
            candidates = tuple(
                task for task in candidates if task.game_family in families
            )
        if config.single_stage is not None:
            candidates = tuple(
                task
                for task in candidates
                if task.single_stage is bool(config.single_stage)
            )
        if config.splits:
            candidates = tuple(
                task for task in candidates if _task_in_any_split(task, config.splits)
            )
        if config.exclude_splits:
            candidates = tuple(
                task
                for task in candidates
                if not _task_in_any_split(task, config.exclude_splits)
            )
        if config.include_validated is not None:
            candidates = tuple(
                task
                for task in candidates
                if task.validated is bool(config.include_validated)
            )
        if config.exclude_validated is not None:
            candidates = tuple(
                task
                for task in candidates
                if task.validated is not bool(config.exclude_validated)
            )
        if config.include_env_ids:
            env_ids = set(config.include_env_ids)
            candidates = tuple(task for task in candidates if task.env_id in env_ids)
        if config.exclude_env_ids:
            env_ids = set(config.exclude_env_ids)
            candidates = tuple(task for task in candidates if task.env_id not in env_ids)
        if config.include_worlds:
            worlds = set(config.include_worlds)
            candidates = tuple(task for task in candidates if task.world in worlds)
        if config.exclude_worlds:
            worlds = set(config.exclude_worlds)
            candidates = tuple(task for task in candidates if task.world not in worlds)
        if config.include_stages:
            stages = set(config.include_stages)
            candidates = tuple(task for task in candidates if task.stage in stages)
        if config.exclude_stages:
            stages = set(config.exclude_stages)
            candidates = tuple(task for task in candidates if task.stage not in stages)

        candidates = tuple(sorted(candidates, key=_task_sort_key))
        if not candidates:
            raise ValueError("task suite has no candidate tasks")
        return candidates


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


def _coerce_task_suite_config(
    config: TaskSuiteConfig | Mapping[str, Any] | None,
) -> TaskSuiteConfig:
    if config is None:
        return TaskSuiteConfig()
    if isinstance(config, TaskSuiteConfig):
        return config
    if isinstance(config, Mapping):
        return TaskSuiteConfig(**dict(config))
    values = {
        field_name: getattr(config, field_name)
        for field_name in TaskSuiteConfig.__dataclass_fields__
        if hasattr(config, field_name)
    }
    return TaskSuiteConfig(**values)


def _validate_task_suite_filters(config: TaskSuiteConfig) -> None:
    valid_splits = {"train", "eval"}
    unknown_splits = set(config.splits) - valid_splits
    unknown_excluded_splits = set(config.exclude_splits) - valid_splits
    if unknown_splits or unknown_excluded_splits:
        names = ", ".join(sorted(unknown_splits | unknown_excluded_splits))
        raise ValueError(f"task suite splits must be train/eval, got: {names}")
    if config.include_validated is not None and config.exclude_validated is not None:
        if bool(config.include_validated) == bool(config.exclude_validated):
            raise ValueError("include_validated and exclude_validated conflict")


def _group_tasks_by_family(tasks: Iterable[MarioTask]) -> dict[str, tuple[MarioTask, ...]]:
    grouped: dict[str, list[MarioTask]] = {}
    for task in tasks:
        grouped.setdefault(task.game_family, []).append(task)
    return {family: tuple(values) for family, values in grouped.items()}


def _task_in_any_split(task: MarioTask, splits: Sequence[str]) -> bool:
    return (
        ("train" in splits and bool(task.train_split))
        or ("eval" in splits and bool(task.eval_split))
    )


def _task_sort_key(task: MarioTask) -> tuple[Any, ...]:
    return (
        task.game_family,
        not task.single_stage,
        task.world if task.world is not None else 0,
        task.stage if task.stage is not None else 0,
        task.env_id,
    )


def _weighted_choice(
    rng: random.Random,
    values: Sequence[str],
    weights: Sequence[float],
) -> str:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("family weights must sum to a positive value")
    threshold = rng.random() * total
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += float(weight)
        if threshold < cumulative:
            return value
    return values[-1]


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


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


def _int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = list(value)
    return tuple(int(item) for item in values)


def _optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"none", "null"}:
        return None
    return _bool(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


def _family_weight_mapping(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(family): float(weight) for family, weight in value.items()}
    weights: dict[str, float] = {}
    for item in _str_tuple(value):
        family, separator, weight = item.partition("=")
        if not separator:
            raise ValueError(
                "family_weights string entries must use FAMILY=WEIGHT"
            )
        weights[family.strip()] = float(weight.strip())
    return weights


__all__ = [
    "MarioTask",
    "TaskFeatureEncoder",
    "TaskFeatures",
    "TaskSuite",
    "TaskSuiteConfig",
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
