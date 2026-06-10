"""Experiment artifact and Trainer configuration helpers."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mario_rl.config import MarioRLConfig, to_dict


@dataclass(frozen=True)
class ExperimentPaths:
    """Stable output paths for one configured experiment."""

    root: Path
    checkpoints: Path
    logs: Path
    videos: Path
    checkpoint: Path
    resolved_config: Path
    train_metrics: Path
    eval_metrics: Path


def experiment_paths(config: MarioRLConfig) -> ExperimentPaths:
    """Return and create the standard artifact directories for ``config``."""
    root = Path(config.save_dir).expanduser() / config.experiment_name
    checkpoints = root / config.train.output_dir_name
    logs = root / "logs"
    videos = root / "videos"
    for path in (checkpoints, logs, videos):
        path.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        root=root,
        checkpoints=checkpoints,
        logs=logs,
        videos=videos,
        checkpoint=checkpoints / config.train.checkpoint_name,
        resolved_config=root / config.train.resolved_config_name,
        train_metrics=root / config.train.metrics_name,
        eval_metrics=root / config.eval.metrics_name,
    )


def checkpoint_path(config: MarioRLConfig, paths: ExperimentPaths | None = None) -> Path:
    """Resolve the checkpoint requested by eval or resume settings."""
    if config.eval.checkpoint:
        return Path(config.eval.checkpoint).expanduser()
    if config.train.checkpoint_path:
        return Path(config.train.checkpoint_path).expanduser()
    paths = paths or experiment_paths(config)
    return paths.checkpoint


def trainer_accelerator(config: MarioRLConfig) -> str:
    """Return a Lightning accelerator string from train/trainer config fields."""
    value = config.train.accelerator or config.trainer.accelerator or "auto"
    normalized = str(value).strip().lower()
    if normalized == "cuda":
        return "gpu"
    return normalized


def trainer_devices(config: MarioRLConfig):
    """Return the Lightning device setting from train/trainer config fields."""
    return config.train.devices if config.train.devices is not None else config.trainer.devices


def write_resolved_config(config: MarioRLConfig, path: Path) -> None:
    """Write a reproducible YAML copy of the resolved config."""
    data = to_dict(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is a declared dependency.
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_train_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Write one stable CSV row with final smoke-training metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "action_set",
        "action_count",
        "native_action_space",
        "reward_transform_mode",
        "reward_missing_total_policy",
        "reward_missing_component_policy",
        "global_step",
        "env_frames",
        "episodes",
        "episode_reward",
        "episode_env_reward",
        "episode_raw_reward",
        "episode_unclipped_reward",
        "episode_clipped_reward",
        "epsilon",
        "loss",
        "learning_rate",
    ]
    row = {name: metrics.get(name, "") for name in fieldnames}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON metrics or command payloads with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
