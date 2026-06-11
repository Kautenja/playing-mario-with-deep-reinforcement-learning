"""Structured Mario task metrics for training and evaluation."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


UNKNOWN_VALUE = "unknown"
_TASK_INFO_KEYS = ("task_id", "game_family", "world", "stage")
_BOOLEAN_INFO_KEYS = ("clear", "death", "timeout")
_PROGRESS_INFO_KEYS = ("progress", "progress_max")
_REWARD_INFO_KEYS = (
    "raw_reward",
    "reward_total_unclipped",
    "reward_total_clipped",
    "reward_components",
)


@dataclass(frozen=True)
class TaskMetricKey:
    """Normalized task identity used for grouping Mario metrics."""

    task_id: str = UNKNOWN_VALUE
    game_family: str = UNKNOWN_VALUE
    world: int | None = None
    stage: int | None = None

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, Any] | None,
        *,
        fallback_env_id: str | None = None,
    ) -> "TaskMetricKey":
        """Resolve task metadata from Gymnasium info with explicit fallbacks."""
        info_map = info if isinstance(info, Mapping) else {}
        task = _task_for_env_id_or_none(
            _optional_str(info_map.get("env_id")) or fallback_env_id
        )

        task_id = _optional_str(info_map.get("task_id"))
        if task_id is None and task is not None:
            task_id = task.task_id
        if task_id is None and fallback_env_id:
            task_id = str(fallback_env_id)

        game_family = _optional_str(info_map.get("game_family"))
        if game_family is None and task is not None:
            game_family = task.game_family

        world = _first_optional_int(
            info_map.get("target_world"),
            info_map.get("world"),
            getattr(task, "world", None) if task is not None else None,
        )
        stage = _first_optional_int(
            info_map.get("target_stage"),
            info_map.get("stage"),
            getattr(task, "stage", None) if task is not None else None,
        )

        return cls(
            task_id=task_id or UNKNOWN_VALUE,
            game_family=game_family or UNKNOWN_VALUE,
            world=world,
            stage=stage,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly task key."""
        return {
            "task_id": self.task_id,
            "game_family": self.game_family,
            "world": self.world,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class StepMetrics:
    """Normalized metrics extracted from one environment step."""

    reward: float
    transformed_reward: float
    raw_reward: float | None
    unclipped_reward: float | None
    clipped_reward: float | None
    clear: bool
    death: bool
    timeout: bool
    terminated: bool
    truncated: bool
    progress: float | None
    progress_max: float | None
    frame_count: int
    task: TaskMetricKey
    reward_components: dict[str, float] = field(default_factory=dict)
    missing_info_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_step(
        cls,
        *,
        reward: float,
        transformed: Any | None = None,
        transformed_reward: float | None = None,
        terminated: bool = False,
        truncated: bool = False,
        info: Mapping[str, Any] | None = None,
        fallback_env_id: str | None = None,
        frame_count: int | None = None,
    ) -> "StepMetrics":
        """Create a typed metric record from a Gymnasium step result."""
        info_map = info if isinstance(info, Mapping) else {}
        env_reward = float(getattr(transformed, "env_reward", reward))
        train_reward = (
            float(transformed_reward)
            if transformed_reward is not None
            else float(getattr(transformed, "training_reward", reward))
        )
        raw_reward = getattr(transformed, "raw_reward", _optional_float(info_map.get("raw_reward")))
        if raw_reward is None and transformed is not None:
            raw_reward = env_reward
        unclipped_reward = getattr(
            transformed,
            "unclipped_reward",
            _optional_float(info_map.get("reward_total_unclipped")),
        )
        clipped_reward = getattr(
            transformed,
            "clipped_reward",
            _optional_float(info_map.get("reward_total_clipped")),
        )
        reward_components = getattr(
            transformed,
            "reward_components",
            _reward_components(info_map.get("reward_components")),
        )

        frames = frame_count
        if frames is None:
            frames = _first_optional_int(info_map.get("frames_skipped"), 1)
        frames = max(int(frames or 1), 1)

        return cls(
            reward=env_reward,
            transformed_reward=train_reward,
            raw_reward=_optional_float(raw_reward),
            unclipped_reward=_optional_float(unclipped_reward),
            clipped_reward=_optional_float(clipped_reward),
            clear=bool(info_map.get("clear", False)),
            death=bool(info_map.get("death", False)),
            timeout=bool(info_map.get("timeout", False)),
            terminated=bool(terminated),
            truncated=bool(truncated),
            progress=_first_optional_float(
                info_map.get("progress"),
                info_map.get("x_pos"),
                info_map.get("position_progress"),
            ),
            progress_max=_first_optional_float(
                info_map.get("progress_max"),
                info_map.get("x_pos_max"),
                info_map.get("position_progress_max"),
            ),
            frame_count=frames,
            task=TaskMetricKey.from_info(info_map, fallback_env_id=fallback_env_id),
            reward_components=dict(reward_components or {}),
            missing_info_counts=_missing_info_counts(info_map),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly step metric record."""
        return {
            "reward": self.reward,
            "transformed_reward": self.transformed_reward,
            "raw_reward": self.raw_reward,
            "unclipped_reward": self.unclipped_reward,
            "clipped_reward": self.clipped_reward,
            "clear": self.clear,
            "death": self.death,
            "timeout": self.timeout,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "progress": self.progress,
            "progress_max": self.progress_max,
            "frame_count": self.frame_count,
            **self.task.to_dict(),
            "reward_components": dict(self.reward_components),
            "missing_info_counts": dict(self.missing_info_counts),
        }


@dataclass(frozen=True)
class EpisodeMetrics:
    """Episode or bounded partial-episode task metrics."""

    episode: int
    complete: bool
    task: TaskMetricKey
    step_count: int
    frame_count: int
    episode_return: float
    transformed_return: float
    raw_return: float | None
    unclipped_return: float | None
    clipped_return: float | None
    clear: bool
    death: bool
    timeout: bool
    terminated: bool
    truncated: bool
    max_progress: float | None
    final_progress: float | None
    reward_component_sums: dict[str, float] = field(default_factory=dict)
    missing_info_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly episode metric record."""
        return {
            "episode": self.episode,
            "complete": self.complete,
            **self.task.to_dict(),
            "steps": self.step_count,
            "frames": self.frame_count,
            "reward": self.episode_return,
            "episode_return": self.episode_return,
            "transformed_return": self.transformed_return,
            "raw_return": self.raw_return,
            "unclipped_return": self.unclipped_return,
            "clipped_return": self.clipped_return,
            "clear": self.clear,
            "death": self.death,
            "timeout": self.timeout,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "max_progress": self.max_progress,
            "final_progress": self.final_progress,
            "reward_component_sums": dict(self.reward_component_sums),
            "missing_info_counts": dict(self.missing_info_counts),
        }


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate Mario metrics for a global or task-group view."""

    name: str
    episode_count: int
    completed_episode_count: int
    step_count: int
    frame_count: int
    episode_return_total: float
    episode_return_mean: float | None
    transformed_return_total: float
    transformed_return_mean: float | None
    raw_return_total: float | None
    raw_return_mean: float | None
    unclipped_return_total: float | None
    unclipped_return_mean: float | None
    clipped_return_total: float | None
    clipped_return_mean: float | None
    clear_count: int
    clear_rate: float | None
    death_count: int
    death_rate: float | None
    timeout_count: int
    timeout_rate: float | None
    truncation_count: int
    truncation_rate: float | None
    max_progress: float | None
    final_progress_mean: float | None
    reward_component_sums: dict[str, float] = field(default_factory=dict)
    missing_info_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_episodes(
        cls,
        episodes: Iterable[EpisodeMetrics],
        *,
        name: str,
    ) -> "AggregateMetrics":
        """Build aggregate metrics from episode records."""
        episode_list = tuple(episodes)
        count = len(episode_list)
        completed_count = sum(1 for item in episode_list if item.complete)
        step_count = sum(item.step_count for item in episode_list)
        frame_count = sum(item.frame_count for item in episode_list)
        episode_return_total = sum(item.episode_return for item in episode_list)
        transformed_return_total = sum(item.transformed_return for item in episode_list)
        raw_values = [item.raw_return for item in episode_list if item.raw_return is not None]
        unclipped_values = [
            item.unclipped_return
            for item in episode_list
            if item.unclipped_return is not None
        ]
        clipped_values = [
            item.clipped_return
            for item in episode_list
            if item.clipped_return is not None
        ]
        clear_count = sum(1 for item in episode_list if item.clear)
        death_count = sum(1 for item in episode_list if item.death)
        timeout_count = sum(1 for item in episode_list if item.timeout)
        truncation_count = sum(1 for item in episode_list if item.truncated)
        progress_values = [
            item.max_progress for item in episode_list if item.max_progress is not None
        ]
        final_progress_values = [
            item.final_progress
            for item in episode_list
            if item.final_progress is not None
        ]
        component_sums: Counter[str] = Counter()
        missing_counts: Counter[str] = Counter()
        for item in episode_list:
            component_sums.update(item.reward_component_sums)
            missing_counts.update(item.missing_info_counts)

        return cls(
            name=name,
            episode_count=count,
            completed_episode_count=completed_count,
            step_count=step_count,
            frame_count=frame_count,
            episode_return_total=float(episode_return_total),
            episode_return_mean=_mean_or_none(
                item.episode_return for item in episode_list
            ),
            transformed_return_total=float(transformed_return_total),
            transformed_return_mean=_mean_or_none(
                item.transformed_return for item in episode_list
            ),
            raw_return_total=_sum_or_none(raw_values),
            raw_return_mean=_mean_or_none(raw_values),
            unclipped_return_total=_sum_or_none(unclipped_values),
            unclipped_return_mean=_mean_or_none(unclipped_values),
            clipped_return_total=_sum_or_none(clipped_values),
            clipped_return_mean=_mean_or_none(clipped_values),
            clear_count=clear_count,
            clear_rate=_rate(clear_count, count),
            death_count=death_count,
            death_rate=_rate(death_count, count),
            timeout_count=timeout_count,
            timeout_rate=_rate(timeout_count, count),
            truncation_count=truncation_count,
            truncation_rate=_rate(truncation_count, count),
            max_progress=max(progress_values) if progress_values else None,
            final_progress_mean=_mean_or_none(final_progress_values),
            reward_component_sums={
                name: float(value) for name, value in sorted(component_sums.items())
            },
            missing_info_counts={
                name: int(value) for name, value in sorted(missing_counts.items())
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate metric record."""
        return {
            "name": self.name,
            "episode_count": self.episode_count,
            "completed_episode_count": self.completed_episode_count,
            "step_count": self.step_count,
            "frame_count": self.frame_count,
            "episode_return_total": self.episode_return_total,
            "episode_return_mean": self.episode_return_mean,
            "transformed_return_total": self.transformed_return_total,
            "transformed_return_mean": self.transformed_return_mean,
            "raw_return_total": self.raw_return_total,
            "raw_return_mean": self.raw_return_mean,
            "unclipped_return_total": self.unclipped_return_total,
            "unclipped_return_mean": self.unclipped_return_mean,
            "clipped_return_total": self.clipped_return_total,
            "clipped_return_mean": self.clipped_return_mean,
            "clear_count": self.clear_count,
            "clear_rate": self.clear_rate,
            "death_count": self.death_count,
            "death_rate": self.death_rate,
            "timeout_count": self.timeout_count,
            "timeout_rate": self.timeout_rate,
            "truncation_count": self.truncation_count,
            "truncation_rate": self.truncation_rate,
            "max_progress": self.max_progress,
            "final_progress_mean": self.final_progress_mean,
            "reward_component_sums": dict(self.reward_component_sums),
            "missing_info_counts": dict(self.missing_info_counts),
        }


class MarioMetricsAccumulator:
    """Accumulate step, episode, and grouped Mario task metrics."""

    def __init__(self, *, default_task_id: str | None = None) -> None:
        self.default_task_id = default_task_id
        self._episodes: list[EpisodeMetrics] = []
        self._active: dict[int, _ActiveEpisode] = {}
        self._next_episode = 0

    @property
    def closed_episodes(self) -> tuple[EpisodeMetrics, ...]:
        """Return completed or explicitly finished episodes."""
        return tuple(self._episodes)

    def start_episode(
        self,
        reset_info: Mapping[str, Any] | None = None,
        *,
        fallback_env_id: str | None = None,
        slot: int = 0,
    ) -> None:
        """Start tracking a new episode from reset info."""
        slot = int(slot)
        active = self._active.get(slot)
        if active is not None and active.step_count > 0:
            self.finish_episode(
                terminated=False,
                truncated=True,
                complete=False,
                slot=slot,
            )
        fallback = fallback_env_id or self.default_task_id
        task = TaskMetricKey.from_info(reset_info, fallback_env_id=fallback)
        self._active[slot] = _ActiveEpisode(episode=self._next_episode, task=task)
        self._next_episode += 1

    def observe_step(
        self,
        *,
        reward: float,
        transformed: Any | None = None,
        transformed_reward: float | None = None,
        terminated: bool = False,
        truncated: bool = False,
        info: Mapping[str, Any] | None = None,
        fallback_env_id: str | None = None,
        frame_count: int | None = None,
        slot: int = 0,
    ) -> StepMetrics:
        """Consume one step and update the active episode."""
        slot = int(slot)
        if slot not in self._active:
            self.start_episode(fallback_env_id=fallback_env_id, slot=slot)
        active = self._active[slot]
        step = StepMetrics.from_step(
            reward=reward,
            transformed=transformed,
            transformed_reward=transformed_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            fallback_env_id=fallback_env_id or self.default_task_id,
            frame_count=frame_count,
        )
        active.apply(step)
        return step

    def finish_episode(
        self,
        *,
        terminated: bool | None = None,
        truncated: bool | None = None,
        complete: bool = True,
        slot: int = 0,
    ) -> EpisodeMetrics | None:
        """Close the active episode and store its aggregate metrics."""
        slot = int(slot)
        active = self._active.pop(slot, None)
        if active is None:
            return None
        episode = active.to_episode_metrics(
            complete=complete,
            terminated=terminated,
            truncated=truncated,
        )
        self._episodes.append(episode)
        return episode

    def active_episode(self, *, slot: int | None = None) -> EpisodeMetrics | None:
        """Return a snapshot of the current partial episode if it has steps."""
        if slot is not None:
            active = self._active.get(int(slot))
            if active is None or active.step_count <= 0:
                return None
            return active.to_episode_metrics(complete=False)
        for active_slot in sorted(self._active):
            active = self._active[active_slot]
            if active.step_count > 0:
                return active.to_episode_metrics(complete=False)
        return None

    def active_episodes(self) -> tuple[EpisodeMetrics, ...]:
        """Return snapshots for all active partial episodes with observed steps."""
        episodes = []
        for slot in sorted(self._active):
            active = self._active[slot]
            if active.step_count > 0:
                episodes.append(active.to_episode_metrics(complete=False))
        return tuple(episodes)

    def episodes(self, *, include_active: bool = False) -> tuple[EpisodeMetrics, ...]:
        """Return closed episodes, optionally including active partial ones."""
        episodes = list(self._episodes)
        if include_active:
            episodes.extend(self.active_episodes())
        return tuple(episodes)

    def global_summary(self, *, include_active: bool = False) -> AggregateMetrics:
        """Return global aggregate metrics."""
        return AggregateMetrics.from_episodes(
            self.episodes(include_active=include_active),
            name="global",
        )

    def to_payload(self, *, include_active: bool = False) -> dict[str, Any]:
        """Return a structured artifact payload for JSON serialization."""
        episodes = self.episodes(include_active=include_active)
        by_family = _group_by(episodes, lambda item: item.task.game_family)
        by_task = _group_by(episodes, lambda item: item.task.task_id)
        active_episodes = self.active_episodes()
        active = active_episodes[0] if active_episodes else None
        return {
            "global": self.global_summary(include_active=include_active).to_dict(),
            "by_game_family": {
                name: AggregateMetrics.from_episodes(items, name=name).to_dict()
                for name, items in sorted(by_family.items())
            },
            "by_task": {
                name: AggregateMetrics.from_episodes(items, name=name).to_dict()
                for name, items in sorted(by_task.items())
            },
            "episodes": [item.to_dict() for item in episodes],
            "closed_episode_count": len(self._episodes),
            "active_episode": active.to_dict() if active is not None else None,
            "active_episodes": [item.to_dict() for item in active_episodes],
            "field_groups": metric_field_groups(),
        }


def metric_field_groups() -> dict[str, tuple[str, ...]]:
    """Return documented metric field groups emitted in JSON artifacts."""
    return {
        "task": _TASK_INFO_KEYS,
        "episode": (
            "episode_return",
            "transformed_return",
            "raw_return",
            "unclipped_return",
            "clipped_return",
            "clear",
            "death",
            "timeout",
            "terminated",
            "truncated",
            "max_progress",
            "final_progress",
            "reward_component_sums",
        ),
        "aggregate": (
            "episode_count",
            "completed_episode_count",
            "step_count",
            "frame_count",
            "episode_return_total",
            "episode_return_mean",
            "transformed_return_total",
            "transformed_return_mean",
            "clear_count",
            "clear_rate",
            "death_count",
            "death_rate",
            "timeout_count",
            "timeout_rate",
            "truncation_count",
            "truncation_rate",
            "max_progress",
            "final_progress_mean",
            "reward_component_sums",
            "missing_info_counts",
        ),
    }


def flatten_global_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten global metrics from a structured payload for one-row CSV output."""
    global_metrics = dict(payload.get("global", {}))
    row = {
        "metric_episode_count": global_metrics.get("episode_count", 0),
        "metric_completed_episode_count": global_metrics.get(
            "completed_episode_count",
            0,
        ),
        "metric_step_count": global_metrics.get("step_count", 0),
        "metric_frame_count": global_metrics.get("frame_count", 0),
        "episode_return_total": global_metrics.get("episode_return_total"),
        "episode_return_mean": global_metrics.get("episode_return_mean"),
        "transformed_return_total": global_metrics.get("transformed_return_total"),
        "transformed_return_mean": global_metrics.get("transformed_return_mean"),
        "raw_return_total": global_metrics.get("raw_return_total"),
        "unclipped_return_total": global_metrics.get("unclipped_return_total"),
        "clipped_return_total": global_metrics.get("clipped_return_total"),
        "clear_count": global_metrics.get("clear_count", 0),
        "clear_rate": global_metrics.get("clear_rate"),
        "death_count": global_metrics.get("death_count", 0),
        "death_rate": global_metrics.get("death_rate"),
        "timeout_count": global_metrics.get("timeout_count", 0),
        "timeout_rate": global_metrics.get("timeout_rate"),
        "truncation_count": global_metrics.get("truncation_count", 0),
        "truncation_rate": global_metrics.get("truncation_rate"),
        "max_progress": global_metrics.get("max_progress"),
        "final_progress_mean": global_metrics.get("final_progress_mean"),
        "reward_component_sums_json": json.dumps(
            global_metrics.get("reward_component_sums", {}),
            sort_keys=True,
        ),
        "missing_info_counts_json": json.dumps(
            global_metrics.get("missing_info_counts", {}),
            sort_keys=True,
        ),
    }
    return row


def write_metrics_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a structured metrics artifact with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a one-row CSV summary for quick spreadsheet inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = flatten_global_metrics(payload)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({name: "" if value is None else value for name, value in row.items()})


@dataclass
class _ActiveEpisode:
    """Mutable state for one episode under construction."""

    episode: int
    task: TaskMetricKey
    step_count: int = 0
    frame_count: int = 0
    episode_return: float = 0.0
    transformed_return: float = 0.0
    raw_return: float | None = None
    unclipped_return: float | None = None
    clipped_return: float | None = None
    clear: bool = False
    death: bool = False
    timeout: bool = False
    terminated: bool = False
    truncated: bool = False
    max_progress: float | None = None
    final_progress: float | None = None
    reward_component_sums: Counter[str] = field(default_factory=Counter)
    missing_info_counts: Counter[str] = field(default_factory=Counter)

    def apply(self, step: StepMetrics) -> None:
        """Add one normalized step to this active episode."""
        if step.task.task_id != UNKNOWN_VALUE or self.task.task_id == UNKNOWN_VALUE:
            self.task = step.task
        self.step_count += 1
        self.frame_count += int(step.frame_count)
        self.episode_return += float(step.reward)
        self.transformed_return += float(step.transformed_reward)
        self.raw_return = _optional_sum(self.raw_return, step.raw_reward)
        self.unclipped_return = _optional_sum(
            self.unclipped_return,
            step.unclipped_reward,
        )
        self.clipped_return = _optional_sum(self.clipped_return, step.clipped_reward)
        self.clear = self.clear or step.clear
        self.death = self.death or step.death
        self.timeout = self.timeout or step.timeout
        self.terminated = self.terminated or step.terminated
        self.truncated = self.truncated or step.truncated
        if step.progress is not None:
            self.final_progress = step.progress
        progress_max = step.progress_max if step.progress_max is not None else step.progress
        if progress_max is not None:
            self.max_progress = (
                progress_max
                if self.max_progress is None
                else max(float(self.max_progress), float(progress_max))
            )
        self.reward_component_sums.update(step.reward_components)
        self.missing_info_counts.update(step.missing_info_counts)

    def to_episode_metrics(
        self,
        *,
        complete: bool,
        terminated: bool | None = None,
        truncated: bool | None = None,
    ) -> EpisodeMetrics:
        """Freeze active state into an episode metric record."""
        is_terminated = self.terminated if terminated is None else bool(terminated)
        is_truncated = self.truncated if truncated is None else bool(truncated)
        return EpisodeMetrics(
            episode=self.episode,
            complete=bool(complete),
            task=self.task,
            step_count=int(self.step_count),
            frame_count=int(self.frame_count),
            episode_return=float(self.episode_return),
            transformed_return=float(self.transformed_return),
            raw_return=self.raw_return,
            unclipped_return=self.unclipped_return,
            clipped_return=self.clipped_return,
            clear=bool(self.clear),
            death=bool(self.death),
            timeout=bool(self.timeout),
            terminated=is_terminated,
            truncated=is_truncated,
            max_progress=self.max_progress,
            final_progress=self.final_progress,
            reward_component_sums={
                name: float(value)
                for name, value in sorted(self.reward_component_sums.items())
            },
            missing_info_counts={
                name: int(value)
                for name, value in sorted(self.missing_info_counts.items())
            },
        )


def _group_by(
    episodes: Iterable[EpisodeMetrics],
    key_fn,
) -> dict[str, tuple[EpisodeMetrics, ...]]:
    groups: defaultdict[str, list[EpisodeMetrics]] = defaultdict(list)
    for item in episodes:
        groups[str(key_fn(item) or UNKNOWN_VALUE)].append(item)
    return {name: tuple(items) for name, items in groups.items()}


def _missing_info_counts(info: Mapping[str, Any]) -> dict[str, int]:
    missing: Counter[str] = Counter()
    for name in (*_BOOLEAN_INFO_KEYS, *_REWARD_INFO_KEYS):
        if name not in info:
            missing[name] += 1
    if "task_id" not in info:
        missing["task_id"] += 1
    if "game_family" not in info:
        missing["game_family"] += 1
    if "world" not in info and "target_world" not in info:
        missing["world"] += 1
    if "stage" not in info and "target_stage" not in info:
        missing["stage"] += 1
    if not any(name in info for name in ("progress", "x_pos", "position_progress")):
        missing["progress"] += 1
    if not any(name in info for name in ("progress_max", "x_pos_max", "position_progress_max")):
        missing["progress_max"] += 1
    return {name: int(value) for name, value in sorted(missing.items())}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _first_optional_float(*values: Any) -> float | None:
    for value in values:
        result = _optional_float(value)
        if result is not None:
            return result
    return None


def _first_optional_int(*values: Any) -> int | None:
    for value in values:
        if value is not None:
            return int(value)
    return None


def _reward_components(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("reward_components must be a mapping")
    return {str(name): float(component) for name, component in value.items()}


def _optional_sum(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    if current is None:
        return float(value)
    return float(current) + float(value)


def _sum_or_none(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(sum(present))


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(sum(present)) / float(len(present))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _task_for_env_id_or_none(env_id: str | None):
    if not env_id:
        return None
    try:
        from mario_rl.envs.tasks import task_for_env_id_or_none
    except Exception:  # pragma: no cover - import fallback for partial installs.
        return None
    return task_for_env_id_or_none(env_id)


__all__ = [
    "AggregateMetrics",
    "EpisodeMetrics",
    "MarioMetricsAccumulator",
    "StepMetrics",
    "TaskMetricKey",
    "UNKNOWN_VALUE",
    "flatten_global_metrics",
    "metric_field_groups",
    "write_metrics_json",
    "write_summary_csv",
]
