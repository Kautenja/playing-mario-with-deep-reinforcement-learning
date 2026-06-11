"""Task metadata and conditioning helpers for the Mario environment surface."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
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
    mode: str = "fixed"
    curriculum_frontier_size: int = 1
    curriculum_mastery_window: int = 5
    curriculum_mastery_min_episodes: int = 3
    curriculum_mastery_clear_rate: float = 0.8
    curriculum_mastery_death_rate: float = 0.25
    curriculum_mastery_progress: float | None = None
    curriculum_lost_levels_prerequisite_family: str = "smb1"
    curriculum_state_path: str | None = None

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
        mode = str(self.mode).strip().lower()
        if mode not in {"fixed", "adaptive"}:
            raise ValueError("task_suite.mode must be 'fixed' or 'adaptive'")
        object.__setattr__(self, "mode", mode)
        frontier_size = int(self.curriculum_frontier_size)
        if frontier_size <= 0:
            raise ValueError("curriculum_frontier_size must be > 0")
        object.__setattr__(self, "curriculum_frontier_size", frontier_size)
        window = int(self.curriculum_mastery_window)
        if window <= 0:
            raise ValueError("curriculum_mastery_window must be > 0")
        object.__setattr__(self, "curriculum_mastery_window", window)
        min_episodes = int(self.curriculum_mastery_min_episodes)
        if min_episodes <= 0:
            raise ValueError("curriculum_mastery_min_episodes must be > 0")
        object.__setattr__(
            self,
            "curriculum_mastery_min_episodes",
            min_episodes,
        )
        clear_rate = float(self.curriculum_mastery_clear_rate)
        death_rate = float(self.curriculum_mastery_death_rate)
        if not 0.0 <= clear_rate <= 1.0:
            raise ValueError("curriculum_mastery_clear_rate must be in [0, 1]")
        if not 0.0 <= death_rate <= 1.0:
            raise ValueError("curriculum_mastery_death_rate must be in [0, 1]")
        object.__setattr__(self, "curriculum_mastery_clear_rate", clear_rate)
        object.__setattr__(self, "curriculum_mastery_death_rate", death_rate)
        if self.curriculum_mastery_progress is not None:
            object.__setattr__(
                self,
                "curriculum_mastery_progress",
                float(self.curriculum_mastery_progress),
            )
        object.__setattr__(
            self,
            "curriculum_lost_levels_prerequisite_family",
            str(self.curriculum_lost_levels_prerequisite_family),
        )
        if self.curriculum_state_path is not None:
            object.__setattr__(
                self,
                "curriculum_state_path",
                str(self.curriculum_state_path),
            )


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

    def observe_episode(self, *_args, **_kwargs) -> None:
        """Accept episode feedback for sampler API parity with adaptive mode."""

    def state_dict(self) -> dict[str, Any]:
        """Return serializable fixed-sampler cursor state."""
        return {
            "mode": "fixed",
            "sample_index": int(self._sample_index),
            "metadata": self.metadata(),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """Restore the fixed-sampler cursor when present."""
        if not isinstance(state, Mapping):
            return
        self._sample_index = int(state.get("sample_index", self._sample_index))

    def metadata(self) -> dict[str, Any]:
        """Return resolved fixed task-suite metadata for artifacts."""
        return {
            "mode": "fixed",
            "enabled": bool(self.config.enabled),
            "candidate_count": len(self.candidates),
            "env_ids": list(self.env_ids),
            "family_counts": self.family_counts,
        }

    def summary_counts(self) -> dict[str, int]:
        """Return curriculum-style counters for fixed sampler artifacts."""
        return {
            "active": len(self.candidates),
            "mastered": 0,
            "locked": 0,
            "retired": 0,
        }

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


@dataclass
class CurriculumTaskProgress:
    """Serializable adaptive-curriculum progress for one task."""

    env_id: str
    task_id: str
    game_family: str
    world: int | None
    stage: int | None
    sampled_episodes: int = 0
    recent_clear_rate: float = 0.0
    recent_death_rate: float = 0.0
    recent_max_progress: float = 0.0
    best_progress: float = 0.0
    mastered: bool = False
    status: str = "locked"
    recent_episodes: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_task(cls, task: MarioTask) -> "CurriculumTaskProgress":
        """Create an empty progress record from registered task metadata."""
        return cls(
            env_id=str(task.env_id),
            task_id=str(task.task_id),
            game_family=str(task.game_family),
            world=_optional_int(getattr(task, "world", None)),
            stage=_optional_int(getattr(task, "stage", None)),
        )

    @classmethod
    def from_dict(
        cls,
        task: MarioTask,
        data: Mapping[str, Any],
    ) -> "CurriculumTaskProgress":
        """Restore a progress record while keeping current task metadata."""
        record = cls.from_task(task)
        record.sampled_episodes = int(
            data.get("sampled_episodes", record.sampled_episodes)
        )
        record.recent_clear_rate = float(
            data.get("recent_clear_rate", record.recent_clear_rate)
        )
        record.recent_death_rate = float(
            data.get("recent_death_rate", record.recent_death_rate)
        )
        record.recent_max_progress = float(
            data.get("recent_max_progress", record.recent_max_progress)
        )
        record.best_progress = float(data.get("best_progress", record.best_progress))
        record.mastered = bool(data.get("mastered", record.mastered))
        record.status = str(data.get("status", record.status))
        record.recent_episodes = tuple(
            dict(item)
            for item in data.get("recent_episodes", ())
            if isinstance(item, Mapping)
        )
        return record

    def record_sample(self) -> None:
        """Record that this task was selected for a training episode."""
        self.sampled_episodes += 1

    def record_episode(
        self,
        *,
        clear: bool,
        death: bool,
        max_progress: float | None,
        config: TaskSuiteConfig,
    ) -> None:
        """Update recent-window progress and mastery from one completed episode."""
        progress = 0.0 if max_progress is None else float(max_progress)
        recent = [
            *self.recent_episodes,
            {
                "clear": bool(clear),
                "death": bool(death),
                "max_progress": progress,
            },
        ]
        window = int(config.curriculum_mastery_window)
        if len(recent) > window:
            recent = recent[-window:]
        self.recent_episodes = tuple(recent)
        count = max(len(recent), 1)
        self.recent_clear_rate = sum(1 for item in recent if item["clear"]) / count
        self.recent_death_rate = sum(1 for item in recent if item["death"]) / count
        self.recent_max_progress = max(
            (float(item["max_progress"]) for item in recent),
            default=0.0,
        )
        self.best_progress = max(self.best_progress, progress)
        progress_threshold = config.curriculum_mastery_progress
        progress_met = (
            progress_threshold is None
            or self.recent_max_progress >= float(progress_threshold)
        )
        if (
            self.sampled_episodes >= int(config.curriculum_mastery_min_episodes)
            and self.recent_clear_rate >= float(config.curriculum_mastery_clear_rate)
            and self.recent_death_rate <= float(config.curriculum_mastery_death_rate)
            and progress_met
        ):
            self.mastered = True

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly record."""
        return {
            "env_id": self.env_id,
            "task_id": self.task_id,
            "game_family": self.game_family,
            "world": self.world,
            "stage": self.stage,
            "sampled_episodes": int(self.sampled_episodes),
            "recent_clear_rate": float(self.recent_clear_rate),
            "recent_death_rate": float(self.recent_death_rate),
            "recent_max_progress": float(self.recent_max_progress),
            "best_progress": float(self.best_progress),
            "mastered": bool(self.mastered),
            "status": self.status,
            "recent_episodes": [dict(item) for item in self.recent_episodes],
        }


class AdaptiveCurriculum:
    """Frontier-based task sampler with explicit serializable progress state."""

    def __init__(
        self,
        config: TaskSuiteConfig | Mapping[str, Any] | None = None,
        *,
        tasks: Iterable[MarioTask] | None = None,
    ) -> None:
        self.config = _coerce_task_suite_config(config)
        base_suite = TaskSuite(self.config, tasks=tasks)
        self.candidates = tuple(sorted(base_suite.candidates, key=_curriculum_sort_key))
        self._task_by_env_id = {str(task.env_id): task for task in self.candidates}
        self._records = {
            str(task.env_id): CurriculumTaskProgress.from_task(task)
            for task in self.candidates
        }
        self._sample_index = 0
        self._episode_task_env_ids: dict[str, str] = {}
        if self.config.curriculum_state_path:
            self.load_state_path(self.config.curriculum_state_path)
        self._refresh_statuses()

    @property
    def env_ids(self) -> tuple[str, ...]:
        """Return candidate environment IDs in curriculum progression order."""
        return tuple(str(task.env_id) for task in self.candidates)

    @property
    def family_counts(self) -> dict[str, int]:
        """Return candidate counts by game family."""
        return _family_counts(self.candidates)

    @property
    def records(self) -> tuple[CurriculumTaskProgress, ...]:
        """Return progress records in curriculum progression order."""
        self._refresh_statuses()
        return tuple(self._records[str(task.env_id)] for task in self.candidates)

    def task_for_index(self, index: int) -> MarioTask:
        """Return the deterministic adaptive task for a zero-based sample index."""
        if index < 0:
            raise ValueError("sample index must be >= 0")
        key = str(int(index))
        if key in self._episode_task_env_ids:
            env_id = self._episode_task_env_ids[key]
            if env_id in self._task_by_env_id:
                return self._task_by_env_id[env_id]

        frontier = self._frontier_for_sampling()
        rng = random.Random(f"{self.config.seed}:adaptive:{int(index)}")
        task = frontier[0] if len(frontier) == 1 else rng.choice(frontier)
        env_id = str(task.env_id)
        self._records[env_id].record_sample()
        self._episode_task_env_ids[key] = env_id
        self._refresh_statuses()
        return task

    def sample(self) -> MarioTask:
        """Return the next deterministic adaptive sample and advance the cursor."""
        task = self.task_for_index(self._sample_index)
        self._sample_index += 1
        return task

    def task_for_episode(self, episode: int) -> MarioTask:
        """Return the active task for an episode number and switch interval."""
        if episode < 0:
            raise ValueError("episode must be >= 0")
        index = int(episode) // int(self.config.switch_interval_episodes)
        return self.task_for_index(index)

    def observe_episode(
        self,
        episode_metrics: Any,
        *,
        env_id: str | None = None,
    ) -> None:
        """Update task progress from completed episode metrics."""
        record_env_id = env_id or _env_id_from_episode_metrics(episode_metrics)
        if record_env_id is None or record_env_id not in self._records:
            return
        self._records[record_env_id].record_episode(
            clear=bool(getattr(episode_metrics, "clear", False)),
            death=bool(getattr(episode_metrics, "death", False)),
            max_progress=getattr(episode_metrics, "max_progress", None),
            config=self.config,
        )
        self._refresh_statuses()

    def load_state_path(self, path: str | Path) -> None:
        """Load curriculum state from a JSON artifact."""
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        self.load_state_dict(payload)

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """Restore serialized progress records and sampling cursor."""
        if not isinstance(state, Mapping):
            return
        payload: Mapping[str, Any] = state
        if isinstance(state.get("curriculum"), Mapping):
            payload = state["curriculum"]
        if isinstance(payload.get("state"), Mapping):
            payload = payload["state"]
        self._sample_index = int(payload.get("sample_index", self._sample_index))
        episode_task_env_ids = payload.get("episode_task_env_ids", {})
        if isinstance(episode_task_env_ids, Mapping):
            self._episode_task_env_ids = {
                str(index): str(env_id)
                for index, env_id in episode_task_env_ids.items()
                if str(env_id) in self._task_by_env_id
            }
        records = payload.get("records", ())
        if isinstance(records, Mapping):
            records = records.values()
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                continue
            env_id = str(raw_record.get("env_id", ""))
            task = self._task_by_env_id.get(env_id)
            if task is None:
                continue
            self._records[env_id] = CurriculumTaskProgress.from_dict(
                task,
                raw_record,
            )
        self._refresh_statuses()

    def state_dict(self) -> dict[str, Any]:
        """Return serializable adaptive-curriculum state."""
        self._refresh_statuses()
        return {
            "mode": "adaptive",
            "sample_index": int(self._sample_index),
            "episode_task_env_ids": dict(sorted(self._episode_task_env_ids.items())),
            "records": [record.to_dict() for record in self.records],
        }

    def metadata(self) -> dict[str, Any]:
        """Return resolved curriculum metadata for artifacts."""
        self._refresh_statuses()
        active = self._active_frontier()
        return {
            "mode": "adaptive",
            "enabled": bool(self.config.enabled),
            "candidate_count": len(self.candidates),
            "frontier_size": int(self.config.curriculum_frontier_size),
            "active_env_ids": [str(task.env_id) for task in active],
            "mastered_env_ids": [
                record.env_id for record in self.records if record.mastered
            ],
            "locked_env_ids": [
                record.env_id for record in self.records if record.status == "locked"
            ],
            "retired_env_ids": [
                record.env_id for record in self.records if record.status == "retired"
            ],
            "family_counts": self.family_counts,
            "mastery": {
                "window": int(self.config.curriculum_mastery_window),
                "min_episodes": int(self.config.curriculum_mastery_min_episodes),
                "clear_rate": float(self.config.curriculum_mastery_clear_rate),
                "death_rate": float(self.config.curriculum_mastery_death_rate),
                "progress": self.config.curriculum_mastery_progress,
            },
            "lost_levels_prerequisite_family": (
                self.config.curriculum_lost_levels_prerequisite_family
            ),
            "tasks": [
                {
                    "env_id": str(task.env_id),
                    "game_family": str(task.game_family),
                    "world": _optional_int(getattr(task, "world", None)),
                    "stage": _optional_int(getattr(task, "stage", None)),
                    "progression_index": index,
                    "status": self._records[str(task.env_id)].status,
                }
                for index, task in enumerate(self.candidates)
            ],
        }

    def payload(self) -> dict[str, Any]:
        """Return metadata plus state for train artifacts."""
        return {
            "metadata": self.metadata(),
            "state": self.state_dict(),
            "counts": self.summary_counts(),
        }

    def summary_counts(self) -> dict[str, int]:
        """Return active, mastered, locked, and retired task counts."""
        self._refresh_statuses()
        counts = {"active": 0, "mastered": 0, "locked": 0, "retired": 0}
        for record in self.records:
            if record.mastered:
                counts["mastered"] += 1
            if record.status in counts:
                counts[record.status] += 1
        return counts

    def _frontier_for_sampling(self) -> tuple[MarioTask, ...]:
        frontier = self._active_frontier()
        if frontier:
            return frontier
        if not self.candidates:
            raise ValueError("adaptive curriculum has no candidate tasks")
        return (self.candidates[-1],)

    def _active_frontier(self) -> tuple[MarioTask, ...]:
        unlocked = []
        for task in self.candidates:
            record = self._records[str(task.env_id)]
            if record.mastered:
                continue
            if self._is_task_locked(task):
                continue
            unlocked.append(task)
        return tuple(unlocked[: int(self.config.curriculum_frontier_size)])

    def _is_task_locked(self, task: MarioTask) -> bool:
        if str(task.game_family) != "lost_levels":
            return False
        prerequisite_family = self.config.curriculum_lost_levels_prerequisite_family
        prerequisites = [
            self._records[str(candidate.env_id)]
            for candidate in self.candidates
            if str(candidate.game_family) == prerequisite_family
        ]
        if not prerequisites:
            return False
        return not all(record.mastered for record in prerequisites)

    def _refresh_statuses(self) -> None:
        active_ids = {str(task.env_id) for task in self._active_frontier()}
        for task in self.candidates:
            record = self._records[str(task.env_id)]
            if record.mastered:
                record.status = "retired"
            elif str(task.env_id) in active_ids:
                record.status = "active"
            else:
                record.status = "locked"


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


def build_task_sampler(
    config: TaskSuiteConfig | Mapping[str, Any] | None,
    *,
    tasks: Iterable[MarioTask] | None = None,
):
    """Build the fixed task suite or adaptive curriculum requested by config."""
    suite_config = _coerce_task_suite_config(config)
    if suite_config.mode == "adaptive":
        return AdaptiveCurriculum(suite_config, tasks=tasks)
    return TaskSuite(suite_config, tasks=tasks)


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


def _curriculum_sort_key(task: MarioTask) -> tuple[Any, ...]:
    family_rank = {
        "smb1": 0,
        "smb2_usa": 1,
        "smb3": 2,
        "lost_levels": 3,
    }.get(str(task.game_family), 50)
    return (
        family_rank,
        not bool(task.single_stage),
        _world_sort_value(getattr(task, "world", None)),
        _world_sort_value(getattr(task, "stage", None)),
        str(task.env_id),
    )


def _world_sort_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, 0)
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _family_counts(tasks: Iterable[MarioTask]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        family = str(task.game_family)
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _env_id_from_episode_metrics(episode_metrics: Any) -> str | None:
    task = getattr(episode_metrics, "task", None)
    if task is None:
        return None
    return _optional_str(getattr(task, "task_id", None))


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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
    "AdaptiveCurriculum",
    "CurriculumTaskProgress",
    "MarioTask",
    "TaskFeatureEncoder",
    "TaskFeatures",
    "TaskSuite",
    "TaskSuiteConfig",
    "UNKNOWN_TASK_VALUE",
    "available_env_ids",
    "available_tasks",
    "build_task_sampler",
    "choose_stage_env_id",
    "encode_task_features",
    "smb3_stage_matrix",
    "task_feature_size",
    "task_for_env_id",
    "task_for_env_id_or_none",
]
