"""Process-local emulator snapshot curriculum support.

Snapshots intentionally keep native emulator state opaque. The library can
capture and restore ``NESEnv.dump_state()`` compatible objects in the current
Python process, but persisted artifacts contain metadata only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mario_rl.config import SnapshotCurriculumConfig


class SnapshotError(RuntimeError):
    """Base class for snapshot curriculum failures."""


class SnapshotUnsupportedError(SnapshotError):
    """Raised when an environment does not expose public snapshot methods."""


class SnapshotCompatibilityError(SnapshotError):
    """Raised when a snapshot is restored into an incompatible environment."""


@dataclass(frozen=True)
class SnapshotMetadata:
    """JSON-safe metadata for one process-local emulator snapshot."""

    snapshot_id: str
    env_id: str
    compatibility_key: str
    action_set: str
    seed_lineage: tuple[str, ...]
    progress: float | None
    episode_step: int | None
    task_id: str
    tags: tuple[str, ...]
    rank_score: float
    rank_reasons: tuple[str, ...]
    capture_index: int

    def to_dict(self) -> dict[str, Any]:
        """Return metadata without native snapshot bytes or observations."""
        return asdict(self)


@dataclass(frozen=True)
class SnapshotEntry:
    """In-process snapshot owner used by training reset paths."""

    metadata: SnapshotMetadata
    native_snapshot: Any
    observation: np.ndarray
    wrapper_state: tuple[dict[str, Any], ...]
    reset_info: dict[str, Any]


class SnapshotLibrary:
    """Ranked, process-local collection of opaque emulator snapshots."""

    def __init__(
        self,
        config: SnapshotCurriculumConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or SnapshotCurriculumConfig()
        self.max_snapshots = _positive_int(self.config.max_snapshots, "max_snapshots")
        self.capture_interval_steps = _positive_int(
            self.config.capture_interval_steps,
            "capture_interval_steps",
        )
        self.sample_probability = _probability(self.config.sample_probability)
        self.rng = random.Random(seed)
        self.entries: list[SnapshotEntry] = []
        self.capture_count = 0
        self.skipped_capture_count = 0
        self.sample_attempt_count = 0
        self.restore_count = 0
        self.fallback_reset_count = 0
        self.incompatible_sample_count = 0

    @property
    def enabled(self) -> bool:
        """Return whether snapshot curriculum behavior is active."""
        return bool(self.config.enabled)

    def maybe_capture(
        self,
        env: Any,
        *,
        observation: Any,
        info: Mapping[str, Any] | None,
        env_id: str,
        action_set: str,
        seed_lineage: Sequence[Any] = (),
        episode_step: int | None = None,
        global_step: int | None = None,
        terminal: bool = False,
    ) -> SnapshotMetadata | None:
        """Capture an eligible training snapshot according to config."""
        if not self.enabled:
            return None
        if terminal:
            self.skipped_capture_count += 1
            return None
        if global_step is not None and int(global_step) % self.capture_interval_steps:
            return None
        info_map = dict(info or {})
        progress = _progress_value(info_map)
        if self.config.min_progress is not None:
            if progress is None or progress < float(self.config.min_progress):
                self.skipped_capture_count += 1
                return None
        try:
            return self.capture(
                env,
                observation=observation,
                info=info_map,
                env_id=env_id,
                action_set=action_set,
                seed_lineage=seed_lineage,
                episode_step=episode_step,
            ).metadata
        except SnapshotUnsupportedError:
            self.skipped_capture_count += 1
            return None

    def capture(
        self,
        env: Any,
        *,
        observation: Any,
        info: Mapping[str, Any] | None,
        env_id: str,
        action_set: str,
        seed_lineage: Sequence[Any] = (),
        episode_step: int | None = None,
    ) -> SnapshotEntry:
        """Capture a process-local snapshot and add it to the library."""
        owner = _state_owner(env)
        if owner is None or not hasattr(owner, "dump_state"):
            raise SnapshotUnsupportedError(
                "environment does not expose dump_state/load_state snapshot methods"
            )
        info_map = dict(info or {})
        native_snapshot = owner.dump_state()
        capture_index = self.capture_count
        metadata = SnapshotMetadata(
            snapshot_id=f"snapshot-{capture_index:06d}",
            env_id=str(env_id),
            compatibility_key=snapshot_compatibility_key(
                env,
                env_id=env_id,
                action_set=action_set,
            ),
            action_set=str(action_set),
            seed_lineage=tuple(str(item) for item in seed_lineage if item is not None),
            progress=_progress_value(info_map),
            episode_step=_episode_step(info_map, episode_step),
            task_id=str(info_map.get("task_id") or env_id),
            tags=_tags_for_info(info_map, self.config.tags),
            rank_score=_rank_score(info_map, self.config.rank_strategy),
            rank_reasons=_rank_reasons(info_map),
            capture_index=capture_index,
        )
        entry = SnapshotEntry(
            metadata=metadata,
            native_snapshot=native_snapshot,
            observation=np.array(observation, copy=True),
            wrapper_state=_capture_wrapper_state(env),
            reset_info=info_map,
        )
        self.entries.append(entry)
        self.capture_count += 1
        self._prune()
        return entry

    def reset_or_restore(
        self,
        env: Any,
        *,
        env_id: str,
        action_set: str,
        seed: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset normally or restore a compatible sampled snapshot start."""
        entry = self.sample_start(env, env_id=env_id, action_set=action_set)
        if entry is None:
            self.fallback_reset_count += 1
            observation, info = env.reset(seed=seed)
            info_map = dict(info or {})
            info_map.setdefault("snapshot_start", False)
            return observation, info_map
        return self.restore(env, entry, seed=seed)

    def sample_start(
        self,
        env: Any,
        *,
        env_id: str,
        action_set: str,
    ) -> SnapshotEntry | None:
        """Return a compatible snapshot reset candidate, or ``None``."""
        if not self.enabled or not self.entries:
            return None
        self.sample_attempt_count += 1
        if self.sample_probability <= 0.0:
            return None
        if self.sample_probability < 1.0 and self.rng.random() > self.sample_probability:
            return None
        key = snapshot_compatibility_key(env, env_id=env_id, action_set=action_set)
        candidates = [entry for entry in self.entries if entry.metadata.compatibility_key == key]
        skipped = len(self.entries) - len(candidates)
        if skipped > 0:
            self.incompatible_sample_count += skipped
        if not candidates:
            return None
        weights = [max(float(entry.metadata.rank_score), 0.001) for entry in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def restore(
        self,
        env: Any,
        entry: SnapshotEntry,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Restore ``entry`` into ``env`` and return the stored pixel observation."""
        expected = entry.metadata.compatibility_key
        actual = snapshot_compatibility_key(
            env,
            env_id=_env_id_from_env(env, entry.metadata.env_id),
            action_set=_action_set_from_env(env, entry.metadata.action_set),
        )
        if actual != expected:
            raise SnapshotCompatibilityError(
                "snapshot compatibility mismatch: "
                f"snapshot {entry.metadata.snapshot_id} was captured for "
                f"{entry.metadata.env_id!r}/{entry.metadata.action_set!r}"
            )
        owner = _state_owner(env)
        if owner is None or not hasattr(owner, "load_state"):
            raise SnapshotUnsupportedError(
                "environment does not expose dump_state/load_state snapshot methods"
            )
        env.reset(seed=seed)
        owner.load_state(entry.native_snapshot)
        _apply_wrapper_state(env, entry.wrapper_state)
        self.restore_count += 1
        info = dict(entry.reset_info)
        info.update(
            {
                "snapshot_start": True,
                "snapshot_id": entry.metadata.snapshot_id,
                "snapshot_rank_score": entry.metadata.rank_score,
                "snapshot_tags": list(entry.metadata.tags),
                "env_id": entry.metadata.env_id,
                "task_id": entry.metadata.task_id,
            }
        )
        return np.array(entry.observation, copy=True), info

    def metadata_entries(self) -> list[dict[str, Any]]:
        """Return JSON-safe metadata for all retained snapshots."""
        return [entry.metadata.to_dict() for entry in self.entries]

    def payload(self) -> dict[str, Any]:
        """Return a JSON-safe artifact payload without native state."""
        return {
            "enabled": self.enabled,
            "config": asdict(self.config),
            "counts": {
                "retained": len(self.entries),
                "captured": self.capture_count,
                "skipped_capture": self.skipped_capture_count,
                "sample_attempts": self.sample_attempt_count,
                "restored": self.restore_count,
                "fallback_resets": self.fallback_reset_count,
                "incompatible_samples": self.incompatible_sample_count,
            },
            "entries": self.metadata_entries(),
            "serialization": {
                "durable_snapshots": False,
                "artifact_contains_native_state": False,
                "artifact_contains_rom_bytes": False,
            },
        }

    def _prune(self) -> None:
        if len(self.entries) <= self.max_snapshots:
            return
        self.entries.sort(
            key=lambda entry: (
                float(entry.metadata.rank_score),
                int(entry.metadata.capture_index),
            ),
            reverse=True,
        )
        del self.entries[self.max_snapshots :]


def snapshot_compatibility_key(env: Any, *, env_id: str, action_set: str) -> str:
    """Return a stable compatibility key for process-local snapshot restores."""
    owner = _state_owner(env)
    rom_path = getattr(owner, "_rom_path", None)
    rom_payload: dict[str, Any]
    if rom_path:
        path = Path(str(rom_path))
        rom_payload = {
            "rom_sha256": _sha256_file(path) if path.is_file() else None,
            "rom_size": path.stat().st_size if path.is_file() else None,
        }
    else:
        rom_payload = {
            "rom_sha256": None,
            "rom_size": None,
        }
    payload = {
        "env_id": str(env_id),
        "action_set": str(action_set),
        "owner_class": _qualified_name(owner),
        "wrapper_stack": [_qualified_name(item) for item in _wrapper_chain(env)],
        "observation_shape": _space_shape(getattr(env, "observation_space", None)),
        "action_space": _space_shape(getattr(env, "action_space", None)),
        "versions": {
            "nes-py": _package_version("nes-py"),
            "gym-super-mario-bros": _package_version("gym-super-mario-bros"),
        },
        **rom_payload,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _state_owner(env: Any) -> Any | None:
    current = getattr(env, "unwrapped", None)
    if current is not None and (
        hasattr(current, "dump_state") or hasattr(current, "load_state")
    ):
        return current
    current = env
    while current is not None:
        if hasattr(current, "dump_state") or hasattr(current, "load_state"):
            return current
        current = getattr(current, "env", None)
    return None


def _env_id_from_env(env: Any, fallback: str) -> str:
    for candidate in (env, _state_owner(env)):
        value = getattr(candidate, "env_id", None)
        if value:
            return str(value)
    return str(fallback)


def _action_set_from_env(env: Any, fallback: str) -> str:
    value = getattr(env, "mario_rl_action_set", None)
    if value:
        return str(value)
    return str(fallback)


def _wrapper_chain(env: Any) -> tuple[Any, ...]:
    wrappers = []
    current = env
    while current is not None:
        wrappers.append(current)
        current = getattr(current, "env", None)
    return tuple(wrappers)


def _capture_wrapper_state(env: Any) -> tuple[dict[str, Any], ...]:
    states: list[dict[str, Any]] = []
    for index, wrapper in enumerate(_wrapper_chain(env)):
        values: dict[str, Any] = {}
        if hasattr(wrapper, "frames"):
            values["frames"] = [np.array(frame, copy=True) for frame in wrapper.frames]
        if hasattr(wrapper, "_obs_buffer"):
            values["obs_buffer"] = [
                np.array(frame, copy=True) for frame in wrapper._obs_buffer
            ]
        for name in (
            "_episode_steps",
            "_steps_since_progress",
            "_best_progress",
            "raw_episode_reward",
            "clipped_episode_reward",
        ):
            if hasattr(wrapper, name):
                values[name] = getattr(wrapper, name)
        if values:
            states.append(
                {
                    "path_index": index,
                    "class": _qualified_name(wrapper),
                    "values": values,
                }
            )
    return tuple(states)


def _apply_wrapper_state(env: Any, states: Sequence[Mapping[str, Any]]) -> None:
    wrappers = _wrapper_chain(env)
    for state in states:
        index = int(state["path_index"])
        if index >= len(wrappers):
            raise SnapshotCompatibilityError("snapshot wrapper stack is shorter than expected")
        wrapper = wrappers[index]
        if _qualified_name(wrapper) != state["class"]:
            raise SnapshotCompatibilityError("snapshot wrapper stack changed")
        values = dict(state.get("values", {}))
        if "frames" in values and hasattr(wrapper, "frames"):
            wrapper.frames.clear()
            for frame in values["frames"]:
                wrapper.frames.append(np.array(frame, copy=True))
        if "obs_buffer" in values and hasattr(wrapper, "_obs_buffer"):
            wrapper._obs_buffer.clear()
            for frame in values["obs_buffer"]:
                wrapper._obs_buffer.append(np.array(frame, copy=True))
        for name in (
            "_episode_steps",
            "_steps_since_progress",
            "_best_progress",
            "raw_episode_reward",
            "clipped_episode_reward",
        ):
            if name in values and hasattr(wrapper, name):
                setattr(wrapper, name, values[name])


def _progress_value(info: Mapping[str, Any]) -> float | None:
    for name in (
        "progress_max",
        "position_progress_max",
        "progress",
        "position_progress",
        "x_pos",
        "x_position",
    ):
        value = info.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _episode_step(info: Mapping[str, Any], explicit: int | None) -> int | None:
    if explicit is not None:
        return int(explicit)
    for name in ("episode_steps", "fake_step", "step", "frame"):
        if info.get(name) is not None:
            return int(info[name])
    return None


def _tags_for_info(
    info: Mapping[str, Any],
    manual_tags: Sequence[str] = (),
) -> tuple[str, ...]:
    tags = {str(tag) for tag in manual_tags if str(tag)}
    if _progress_value(info) is not None:
        tags.add("progress")
    if any(bool(info.get(name)) for name in ("death", "is_dying", "death_proximity")):
        tags.add("death-proximity")
    repeated_failure = _first_number(
        info,
        "no_progress_steps",
        "failure_count",
        "repeated_failure_count",
    )
    if repeated_failure is not None and repeated_failure > 0:
        tags.add("repeated-failure")
    if any(bool(info.get(name)) for name in ("clear", "flag_get", "level_complete", "clear_proximity")):
        tags.add("clear-proximity")
    return tuple(sorted(tags))


def _rank_score(info: Mapping[str, Any], strategy: str) -> float:
    strategy = str(strategy or "progress").strip().lower()
    progress = _progress_value(info) or 0.0
    score = float(progress if strategy == "progress" else progress)
    tags = set(_tags_for_info(info))
    if "death-proximity" in tags:
        score += 10.0
    if "repeated-failure" in tags:
        score += 5.0
    if "clear-proximity" in tags:
        score += 20.0
    return float(score)


def _rank_reasons(info: Mapping[str, Any]) -> tuple[str, ...]:
    reasons = []
    if _progress_value(info) is not None:
        reasons.append("progress")
    for tag in _tags_for_info(info):
        if tag not in reasons:
            reasons.append(tag)
    return tuple(reasons)


def _first_number(info: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = info.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qualified_name(value: Any) -> str:
    if value is None:
        return "None"
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _space_shape(space: Any) -> Any:
    if space is None:
        return None
    if hasattr(space, "shape"):
        return tuple(int(item) for item in space.shape)
    if hasattr(space, "n"):
        return int(space.n)
    return str(space)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"snapshot {name} must be > 0")
    return value


def _probability(value: float) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("snapshot sample_probability must be between 0 and 1")
    return value


__all__ = [
    "SnapshotCompatibilityError",
    "SnapshotCurriculumConfig",
    "SnapshotEntry",
    "SnapshotError",
    "SnapshotLibrary",
    "SnapshotMetadata",
    "SnapshotUnsupportedError",
    "snapshot_compatibility_key",
]
