"""Evaluation matrix construction and runner for Mario RL checkpoints."""
from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationMatrixConfig:
    """Filters, seed expansion, and artifact names for matrix evaluation."""

    enabled: bool = False
    game_families: tuple[str, ...] = ()
    single_stage: bool | None = True
    splits: tuple[str, ...] = ("eval",)
    exclude_splits: tuple[str, ...] = ()
    include_validated: bool | None = True
    exclude_validated: bool | None = None
    include_aliases: bool = False
    include_env_ids: tuple[str, ...] = ()
    exclude_env_ids: tuple[str, ...] = ()
    max_tasks: int | None = None
    include_smb3_catalog: bool = False
    seeds: tuple[int, ...] = ()
    seed: int | None = 123
    seed_count: int = 1
    episodes_per_task: int = 1
    summary_name: str = "eval-matrix-summary.json"
    table_name: str = "eval-matrix-episodes.csv"
    video_enabled: bool = False
    video_dir: str | None = None
    video_length: int = 0
    video_name_prefix: str = "eval-matrix"

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
        if self.max_tasks is not None:
            max_tasks = int(self.max_tasks)
            if max_tasks <= 0:
                raise ValueError("evaluation_matrix.max_tasks must be > 0")
            object.__setattr__(self, "max_tasks", max_tasks)
        object.__setattr__(
            self,
            "include_smb3_catalog",
            _bool(self.include_smb3_catalog),
        )
        object.__setattr__(self, "seeds", _int_tuple(self.seeds))
        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))
        seed_count = int(self.seed_count)
        if seed_count <= 0:
            raise ValueError("evaluation_matrix.seed_count must be > 0")
        object.__setattr__(self, "seed_count", seed_count)
        episodes = int(self.episodes_per_task)
        if episodes <= 0:
            raise ValueError("evaluation_matrix.episodes_per_task must be > 0")
        object.__setattr__(self, "episodes_per_task", episodes)
        object.__setattr__(self, "summary_name", str(self.summary_name))
        object.__setattr__(self, "table_name", str(self.table_name))
        object.__setattr__(self, "video_enabled", _bool(self.video_enabled))
        if self.video_dir is not None:
            object.__setattr__(self, "video_dir", str(self.video_dir))
        object.__setattr__(self, "video_length", int(self.video_length))
        object.__setattr__(self, "video_name_prefix", str(self.video_name_prefix))


@dataclass(frozen=True)
class EvaluationMatrixEntry:
    """One runnable task or catalog-only metadata row in an evaluation matrix."""

    env_id: str
    task_id: str
    game: str
    game_family: str
    version: int
    rom_mode: str
    world: int | None = None
    stage: int | None = None
    world_label: str | None = None
    single_stage: bool = False
    train_split: bool = True
    eval_split: bool = True
    validated: bool = True
    registered: bool = True
    runnable: bool = True
    source: str = "registered"
    reason: str | None = None

    @classmethod
    def from_task(cls, task: Any) -> "EvaluationMatrixEntry":
        """Create a runnable entry from a ``gym-super-mario-bros`` task."""
        return cls(
            env_id=str(task.env_id),
            task_id=str(task.task_id),
            game=str(task.game),
            game_family=str(task.game_family),
            version=int(task.version),
            rom_mode=str(task.rom_mode),
            world=_optional_int(task.world),
            stage=_optional_int(task.stage),
            world_label=_optional_str(task.world_label),
            single_stage=bool(task.single_stage),
            train_split=bool(task.train_split),
            eval_split=bool(task.eval_split),
            validated=bool(task.validated),
            registered=True,
            runnable=True,
            source="registered",
        )

    @classmethod
    def from_smb3_catalog(cls, stage: Any, *, registered: bool) -> "EvaluationMatrixEntry":
        """Create a metadata entry for the SMB3 stage catalog."""
        return cls(
            env_id=str(stage.env_id),
            task_id=str(stage.env_id),
            game="smb3",
            game_family="smb3",
            version=0,
            rom_mode="vanilla",
            world=int(stage.world),
            stage=int(stage.stage),
            world_label=str(stage.world_label),
            single_stage=True,
            validated=bool(stage.validated),
            registered=bool(registered),
            runnable=False,
            source="smb3_catalog",
            reason=(
                "registered separately in selected tasks"
                if registered
                else "unregistered SMB3 catalog stage"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row."""
        return {
            "env_id": self.env_id,
            "task_id": self.task_id,
            "game": self.game,
            "game_family": self.game_family,
            "version": self.version,
            "rom_mode": self.rom_mode,
            "world": self.world,
            "stage": self.stage,
            "world_label": self.world_label,
            "single_stage": self.single_stage,
            "train_split": self.train_split,
            "eval_split": self.eval_split,
            "validated": self.validated,
            "registered": self.registered,
            "runnable": self.runnable,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvaluationMatrix:
    """Runnable tasks plus optional catalog-only metadata."""

    tasks: tuple[EvaluationMatrixEntry, ...]
    metadata: tuple[EvaluationMatrixEntry, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly matrix description."""
        return {
            "task_count": len(self.tasks),
            "metadata_count": len(self.metadata),
            "tasks": [entry.to_dict() for entry in self.tasks],
            "metadata": [entry.to_dict() for entry in self.metadata],
            "family_counts": _family_counts(self.tasks),
            "metadata_family_counts": _family_counts(self.metadata),
        }


class ConstantPolicy:
    """Small fake policy useful for no-ROM unit tests."""

    def __init__(self, action: int = 0) -> None:
        self.action = int(action)

    def reset(self, *, task: EvaluationMatrixEntry, seed: int) -> None:
        """Reset policy state for a new matrix episode."""

    def act(
        self,
        observation: Any,
        *,
        task: EvaluationMatrixEntry,
        deterministic: bool,
    ) -> int:
        """Return the configured action."""
        return self.action

    def close(self) -> None:
        """Release policy resources."""


EnvFactory = Callable[[Any], Any]
PolicyFactory = Callable[[Any], Any]


def build_evaluation_matrix(
    config: EvaluationMatrixConfig | Mapping[str, Any] | None = None,
    *,
    tasks: Iterable[Any] | None = None,
) -> EvaluationMatrix:
    """Build runnable matrix entries from task filters and optional metadata."""
    from mario_rl.envs.tasks import TaskSuite, TaskSuiteConfig, available_tasks, smb3_stage_matrix

    matrix_config = _coerce_matrix_config(config)
    catalog = tuple(tasks) if tasks is not None else available_tasks(
        include_aliases=matrix_config.include_aliases
    )
    _validate_explicit_env_ids(matrix_config, catalog)
    suite_config = TaskSuiteConfig(
        enabled=True,
        game_families=matrix_config.game_families,
        single_stage=matrix_config.single_stage,
        splits=matrix_config.splits,
        exclude_splits=matrix_config.exclude_splits,
        include_validated=matrix_config.include_validated,
        exclude_validated=matrix_config.exclude_validated,
        include_aliases=matrix_config.include_aliases,
        include_env_ids=matrix_config.include_env_ids,
        exclude_env_ids=matrix_config.exclude_env_ids,
    )
    suite = TaskSuite(suite_config, tasks=catalog)
    selected = suite.candidates
    if matrix_config.max_tasks is not None:
        selected = selected[: int(matrix_config.max_tasks)]

    tasks_by_env_id = {str(task.env_id): task for task in catalog}
    metadata: tuple[EvaluationMatrixEntry, ...] = ()
    if matrix_config.include_smb3_catalog:
        metadata = tuple(
            EvaluationMatrixEntry.from_smb3_catalog(
                stage,
                registered=str(stage.env_id) in tasks_by_env_id,
            )
            for stage in smb3_stage_matrix()
        )
    return EvaluationMatrix(
        tasks=tuple(EvaluationMatrixEntry.from_task(task) for task in selected),
        metadata=metadata,
    )


def expand_evaluation_seeds(
    config: EvaluationMatrixConfig | Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """Return explicit or deterministic sequential matrix seeds."""
    matrix_config = _coerce_matrix_config(config)
    if matrix_config.seeds:
        return tuple(int(seed) for seed in matrix_config.seeds)
    if matrix_config.seed is None:
        raise ValueError("evaluation_matrix.seed is required when seeds is empty")
    return tuple(
        int(matrix_config.seed) + index for index in range(int(matrix_config.seed_count))
    )


def matrix_video_prefix(
    config: EvaluationMatrixConfig | Mapping[str, Any],
    entry: EvaluationMatrixEntry,
    *,
    seed: int,
    episode: int,
) -> str:
    """Return the stable video filename prefix for one matrix episode."""
    matrix_config = _coerce_matrix_config(config)
    task_slug = _slug(entry.env_id)
    return (
        f"{matrix_config.video_name_prefix}-{task_slug}-"
        f"seed-{int(seed)}-episode-{int(episode)}"
    )


def run_evaluation_matrix(
    config: Any,
    *,
    env_factory: EnvFactory | None = None,
    policy_factory: PolicyFactory | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint or injected policy over the configured matrix."""
    from mario_rl.config import action_space_summary, with_resolved_model_num_actions
    from mario_rl.lightning.artifacts import checkpoint_path, experiment_paths, write_json
    from mario_rl.metrics import MarioMetricsAccumulator
    from mario_rl.rewards import RewardTransformer

    config = with_resolved_model_num_actions(config)
    paths = experiment_paths(config)
    matrix_config = _coerce_matrix_config(config.evaluation_matrix)
    matrix = build_evaluation_matrix(matrix_config)
    seeds = expand_evaluation_seeds(matrix_config)
    checkpoint = checkpoint_path(config, paths)
    policy = (
        policy_factory(config)
        if policy_factory is not None
        else _checkpoint_policy(config, checkpoint=checkpoint)
    )
    metrics = MarioMetricsAccumulator(default_task_id=config.env.id)
    reward_transformer = RewardTransformer(config.reward_transform)
    rows: list[dict[str, Any]] = []

    try:
        for task_index, entry in enumerate(matrix.tasks):
            for seed in seeds:
                for episode in range(int(matrix_config.episodes_per_task)):
                    episode_row = _run_matrix_episode(
                        config,
                        matrix_config,
                        entry,
                        task_index=task_index,
                        seed=seed,
                        matrix_episode=episode,
                        env_factory=env_factory,
                        policy=policy,
                        metrics=metrics,
                        reward_transformer=reward_transformer,
                    )
                    rows.append(episode_row)
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    metrics_payload = metrics.to_payload(include_active=False)
    summary_path = paths.root / matrix_config.summary_name
    table_path = paths.root / matrix_config.table_name
    payload = {
        "command": "eval-matrix",
        **action_space_summary(config),
        "checkpoint": str(checkpoint),
        "summary_path": str(summary_path),
        "table_path": str(table_path),
        "seeds": list(seeds),
        "episodes_per_task": int(matrix_config.episodes_per_task),
        "row_count": len(rows),
        "matrix": matrix.to_dict(),
        **metrics_payload,
    }
    write_episode_table(table_path, rows)
    write_json(summary_path, payload)
    return payload


def write_episode_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one CSV row per evaluated task/seed/episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_index",
        "env_id",
        "task_id",
        "game",
        "game_family",
        "world",
        "stage",
        "single_stage",
        "validated",
        "seed",
        "matrix_episode",
        "episode",
        "complete",
        "steps",
        "frames",
        "reward",
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
        "reward_component_sums_json",
        "missing_info_counts_json",
        "video_prefix",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _run_matrix_episode(
    config: Any,
    matrix_config: EvaluationMatrixConfig,
    entry: EvaluationMatrixEntry,
    *,
    task_index: int,
    seed: int,
    matrix_episode: int,
    env_factory: EnvFactory | None,
    policy: Any,
    metrics: Any,
    reward_transformer: Any,
) -> dict[str, Any]:
    episode_config = _episode_config(
        config,
        matrix_config,
        entry,
        seed=seed,
        episode=matrix_episode,
    )
    if env_factory is None:
        from mario_rl.envs import make_env

        env = make_env(config=episode_config.env.to_mario_env_config())
    else:
        env = env_factory(episode_config)
    try:
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy(task=entry, seed=seed)
        state, reset_info = env.reset(seed=seed)
        metrics.start_episode(
            reset_info if isinstance(reset_info, Mapping) else None,
            fallback_env_id=entry.env_id,
        )
        steps = 0
        terminated = False
        truncated = False
        while steps < int(config.eval.max_steps):
            action = policy.act(
                state,
                task=entry,
                deterministic=bool(config.eval.deterministic),
            )
            state, reward, terminated, truncated, info = env.step(action)
            info_map = info if isinstance(info, Mapping) else None
            transformed = reward_transformer.transform(float(reward), info_map)
            frames = int(info.get("frames_skipped", 1)) if isinstance(info, Mapping) else 1
            metrics.observe_step(
                reward=float(reward),
                transformed=transformed,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info_map,
                fallback_env_id=entry.env_id,
                frame_count=max(frames, 1),
            )
            steps += 1
            if terminated or truncated:
                break
        limit_truncated = (
            steps >= int(config.eval.max_steps)
            and not bool(terminated)
            and not bool(truncated)
        )
        episode = metrics.finish_episode(
            terminated=bool(terminated),
            truncated=bool(truncated or limit_truncated),
        )
        if episode is None:
            raise RuntimeError("matrix episode produced no metrics")
        episode_payload = episode.to_dict()
        return {
            **entry.to_dict(),
            **episode_payload,
            "task_index": int(task_index),
            "seed": int(seed),
            "matrix_episode": int(matrix_episode),
            "reward_component_sums_json": json.dumps(
                episode_payload.get("reward_component_sums", {}),
                sort_keys=True,
            ),
            "missing_info_counts_json": json.dumps(
                episode_payload.get("missing_info_counts", {}),
                sort_keys=True,
            ),
            "video_prefix": matrix_video_prefix(
                matrix_config,
                entry,
                seed=seed,
                episode=matrix_episode,
            )
            if matrix_config.video_enabled
            else "",
        }
    finally:
        env.close()


def _episode_config(
    config: Any,
    matrix_config: EvaluationMatrixConfig,
    entry: EvaluationMatrixEntry,
    *,
    seed: int,
    episode: int,
) -> Any:
    env_updates = {
        "id": entry.env_id,
        "seed": int(seed),
    }
    if matrix_config.video_enabled:
        video_dir = matrix_config.video_dir
        if video_dir is None:
            from mario_rl.lightning.artifacts import experiment_paths

            video_dir = str(experiment_paths(config).videos / "eval-matrix")
        env_updates.update(
            {
                "render_mode": config.env.render_mode or "rgb_array",
                "video_enabled": True,
                "video_dir": video_dir,
                "video_length": int(matrix_config.video_length),
                "video_name_prefix": matrix_video_prefix(
                    matrix_config,
                    entry,
                    seed=seed,
                    episode=episode,
                ),
            }
        )
    return replace(config, env=replace(config.env, **env_updates))


def _checkpoint_policy(config: Any, *, checkpoint: Path) -> Any:
    algorithm = str(getattr(config.train, "algorithm", "dqn")).strip().lower()
    if algorithm in {"dqn", "deep_q", "deep_q_network"}:
        return _DQNCheckpointPolicy(config, checkpoint=checkpoint)
    if algorithm in {"ppo", "actor_critic", "recurrent_actor_critic"}:
        return _PPOCheckpointPolicy(config, checkpoint=checkpoint)
    raise ValueError(f"unsupported evaluation algorithm: {config.train.algorithm!r}")


class _DQNCheckpointPolicy:
    def __init__(self, config: Any, *, checkpoint: Path) -> None:
        import torch

        from mario_rl.envs import TaskFeatureEncoder
        from mario_rl.lightning.module import DQNLightningModule
        from mario_rl.schedules import EpsilonGreedyActionSelector

        self._torch = torch
        self.module = DQNLightningModule.load_from_checkpoint(
            str(checkpoint),
            config=config,
            map_location="cpu",
        )
        self.module.eval()
        self.network = self.module.q_network
        self.selector = EpsilonGreedyActionSelector(
            num_actions=config.model.num_actions,
            seed=config.trainer.seed if config.trainer.seed is not None else config.env.seed,
        )
        self.encoder = None
        self.task_feature_size = int(getattr(self.network, "task_feature_size", 0))
        if self.task_feature_size > 0:
            self.encoder = TaskFeatureEncoder()
            if self.encoder.feature_size != self.task_feature_size:
                raise ValueError(
                    "checkpoint task feature size "
                    f"{self.task_feature_size} does not match encoder size "
                    f"{self.encoder.feature_size}"
                )

    def reset(self, *, task: EvaluationMatrixEntry, seed: int) -> None:
        """DQN evaluation has no recurrent state."""

    def act(
        self,
        observation: Any,
        *,
        task: EvaluationMatrixEntry,
        deterministic: bool,
    ) -> int:
        state = self._torch.as_tensor(observation).unsqueeze(0)
        task_features = self._task_features(task)
        with self._torch.no_grad():
            q_values = self.network(state, task_features).squeeze(0).cpu()
        return self.selector.select(q_values, epsilon=0.0, deterministic=deterministic)

    def close(self) -> None:
        """Release checkpoint policy resources."""

    def _task_features(self, task: EvaluationMatrixEntry):
        if self.encoder is None:
            return None
        return self.encoder.encode_env_id(task.env_id).to_tensor().unsqueeze(0)


class _PPOCheckpointPolicy:
    def __init__(self, config: Any, *, checkpoint: Path) -> None:
        import torch
        from torch.distributions import Categorical

        from mario_rl.envs import TaskFeatureEncoder
        from mario_rl.lightning.ppo_module import PPOLightningModule

        self._torch = torch
        self._categorical = Categorical
        self.module = PPOLightningModule.load_from_checkpoint(
            str(checkpoint),
            config=config,
            map_location="cpu",
        )
        self.module.eval()
        self.policy = self.module.policy
        self.hidden_state = None
        self.encoder = None
        self.task_feature_size = int(getattr(self.policy, "task_feature_size", 0))
        if self.task_feature_size > 0:
            self.encoder = TaskFeatureEncoder()
            if self.encoder.feature_size != self.task_feature_size:
                raise ValueError(
                    "checkpoint task feature size "
                    f"{self.task_feature_size} does not match encoder size "
                    f"{self.encoder.feature_size}"
                )

    def reset(self, *, task: EvaluationMatrixEntry, seed: int) -> None:
        self._torch.manual_seed(int(seed))
        self.hidden_state = self.policy.initial_state(1, device="cpu")

    def act(
        self,
        observation: Any,
        *,
        task: EvaluationMatrixEntry,
        deterministic: bool,
    ) -> int:
        if self.hidden_state is None:
            self.hidden_state = self.policy.initial_state(1, device="cpu")
        state = self._torch.as_tensor(observation).unsqueeze(0)
        with self._torch.no_grad():
            output = self.policy(state, self.hidden_state, self._task_features(task))
        self.hidden_state = output.hidden_state.detach()
        if deterministic:
            return int(output.policy_logits.squeeze(0).argmax().item())
        distribution = self._categorical(logits=output.policy_logits.squeeze(0))
        return int(distribution.sample().item())

    def close(self) -> None:
        """Release checkpoint policy resources."""

    def _task_features(self, task: EvaluationMatrixEntry):
        if self.encoder is None:
            return None
        return self.encoder.encode_env_id(task.env_id).to_tensor().unsqueeze(0)


def _coerce_matrix_config(
    config: EvaluationMatrixConfig | Mapping[str, Any] | None,
) -> EvaluationMatrixConfig:
    if config is None:
        return EvaluationMatrixConfig()
    if isinstance(config, EvaluationMatrixConfig):
        return config
    if isinstance(config, Mapping):
        return EvaluationMatrixConfig(**dict(config))
    values = {
        field_name: getattr(config, field_name)
        for field_name in EvaluationMatrixConfig.__dataclass_fields__
        if hasattr(config, field_name)
    }
    return EvaluationMatrixConfig(**values)


def _validate_explicit_env_ids(
    config: EvaluationMatrixConfig,
    tasks: Sequence[Any],
) -> None:
    known = {str(task.env_id) for task in tasks}
    requested = set(config.include_env_ids) | set(config.exclude_env_ids)
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown evaluation matrix env ID(s): {', '.join(unknown)}")


def _family_counts(entries: Sequence[EvaluationMatrixEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.game_family] = counts.get(entry.game_family, 0) + 1
    return dict(sorted(counts.items()))


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return slug or "task"


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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


__all__ = [
    "ConstantPolicy",
    "EvaluationMatrix",
    "EvaluationMatrixConfig",
    "EvaluationMatrixEntry",
    "build_evaluation_matrix",
    "expand_evaluation_seeds",
    "matrix_video_prefix",
    "run_evaluation_matrix",
    "write_episode_table",
]
