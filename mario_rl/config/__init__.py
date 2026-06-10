"""Typed experiment configuration and CLI helpers for Mario RL."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

from mario_rl.auxiliary import AuxiliaryLossConfig
from mario_rl.evaluation_matrix import EvaluationMatrixConfig
from mario_rl.envs.tasks import TaskSuiteConfig
from mario_rl.rewards import RewardTransformConfig

try:  # pragma: no cover - exercised when optional dependency is installed.
    from jsonargparse import ArgumentParser as _ArgumentParser
except ImportError:  # pragma: no cover - local test fallback.
    _ArgumentParser = argparse.ArgumentParser


AUTO_NUM_ACTIONS = "auto"


@dataclass(frozen=True)
class TrainerConfig:
    """Lightning trainer and device placement settings."""

    accelerator: str = "auto"
    devices: int | str | None = "auto"
    precision: str = "32-true"
    deterministic: bool = True
    seed: int | None = 123
    enable_progress_bar: bool = True


@dataclass(frozen=True)
class EnvConfig:
    """Gymnasium environment, preprocessing, smoke, and video settings."""

    id: str = "SuperMarioBros-1-1-v0"
    render_mode: str | None = None
    action_set: str = "simple"
    seed: int | None = 123
    image_size: tuple[int, int] = (84, 84)
    frame_stack: int | None = 4
    reward_clipping: bool = False
    frame_skip: int | None = 4
    preprocess: bool = True
    grayscale: bool = True
    channel_first: bool = True
    interpolation: str = "area"
    record_statistics: bool = True
    video_enabled: bool = False
    video_dir: str | None = None
    video_length: int = 0
    video_name_prefix: str = "mario-rl"
    max_smoke_steps: int = 128

    def to_mario_env_config(self):
        """Convert to the existing environment factory config."""
        from mario_rl.envs.config import MarioEnvConfig

        return MarioEnvConfig(
            env_id=self.id,
            render_mode=self.render_mode,
            seed=self.seed,
            action_set=self.action_set,
            preprocess=self.preprocess,
            frame_skip=self.frame_skip,
            image_size=self.image_size,
            interpolation=self.interpolation,
            grayscale=self.grayscale,
            channel_first=self.channel_first,
            frame_stack=self.frame_stack,
            clip_rewards=self.reward_clipping,
            record_statistics=self.record_statistics,
            video_dir=self.video_dir if self.video_enabled else None,
            video_length=self.video_length,
            video_name_prefix=self.video_name_prefix,
        )


@dataclass(frozen=True)
class ReplayConfig:
    """Replay-buffer shape, dtype, and sampling settings."""

    capacity: int = 100_000
    batch_size: int = 32
    warmup: int = 1_000
    prioritized: bool = False
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    sample_dtype: str = "uint8"
    state_shape: tuple[int, int, int] = (4, 84, 84)
    store_reward_info: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Model and optimizer settings for DQN and actor-critic paths."""

    architecture: str = "dqn"
    input_channels: int = 4
    hidden_size: int = 512
    recurrent_hidden_size: int = 256
    task_embedding_size: int = 32
    optimizer: str = "adam"
    learning_rate: float = 0.00025
    discount_factor: float = 0.99
    double_dqn: bool = True
    target_update_frequency: int = 10_000
    compile: bool = False
    num_actions: int | str = AUTO_NUM_ACTIONS
    task_conditioning: bool = False
    task_feature_size: int = 0


@dataclass(frozen=True)
class PPOConfig:
    """Rollout and PPO optimization settings for recurrent actor-critic."""

    rollout_steps: int = 32
    minibatch_size: int = 16
    epochs: int = 2
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    normalize_advantages: bool = True
    max_grad_norm: float | None = 0.5


@dataclass(frozen=True)
class EpsilonConfig:
    """Exploration schedule settings."""

    start: float = 1.0
    final: float = 0.1
    decay_frames: int = 1_000_000


@dataclass(frozen=True)
class TrainConfig:
    """Training loop limits, logging, checkpoint, and artifact names."""

    algorithm: str = "dqn"
    max_frames: int = 1_000_000
    max_steps: int = 250_000
    fast_dev_run: bool = False
    log_interval: int = 1_000
    checkpoint_path: str | None = None
    accelerator: str = "auto"
    devices: int | str | None = "auto"
    output_dir_name: str = "checkpoints"
    checkpoint_name: str = "last.ckpt"
    resolved_config_name: str = "resolved-config.yaml"
    metrics_name: str = "train-metrics.csv"


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation and play settings."""

    checkpoint: str | None = None
    episodes: int = 1
    max_steps: int = 1_000
    deterministic: bool = True
    metrics_name: str = "eval-metrics.json"
    video_name: str = "eval.mp4"


@dataclass(frozen=True)
class MarioRLConfig:
    """Single typed config tree for the modern Mario learner."""

    experiment_name: str = "smb_dqn_fast_dev"
    save_dir: str = "runs"
    trainer: TrainerConfig = TrainerConfig()
    env: EnvConfig = EnvConfig()
    task_suite: TaskSuiteConfig = field(default_factory=TaskSuiteConfig)
    reward_transform: RewardTransformConfig = field(default_factory=RewardTransformConfig)
    auxiliary: AuxiliaryLossConfig = field(default_factory=AuxiliaryLossConfig)
    evaluation_matrix: EvaluationMatrixConfig = field(default_factory=EvaluationMatrixConfig)
    replay: ReplayConfig = ReplayConfig()
    model: ModelConfig = ModelConfig()
    ppo: PPOConfig = PPOConfig()
    epsilon: EpsilonConfig = EpsilonConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()


_CONFIG_PACKAGE = "mario_rl.config"
_CONFIG_DATA = "data"
_SECTIONS = {
    "trainer": TrainerConfig,
    "env": EnvConfig,
    "task_suite": TaskSuiteConfig,
    "reward_transform": RewardTransformConfig,
    "auxiliary": AuxiliaryLossConfig,
    "evaluation_matrix": EvaluationMatrixConfig,
    "replay": ReplayConfig,
    "model": ModelConfig,
    "ppo": PPOConfig,
    "epsilon": EpsilonConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}
_TOP_LEVEL_FIELDS = {"experiment_name", "save_dir"}
_UNSET = object()


def available_configs() -> tuple[str, ...]:
    """Return packaged config names sorted by file stem."""
    data_dir = resources.files(_CONFIG_PACKAGE).joinpath(_CONFIG_DATA)
    names = [
        path.name.removesuffix(".yaml")
        for path in data_dir.iterdir()
        if path.name.endswith((".yaml", ".yml"))
    ]
    return tuple(sorted(names))


def config_path(name: str) -> Path:
    """Return an absolute path for a packaged config name."""
    data_dir = resources.files(_CONFIG_PACKAGE).joinpath(_CONFIG_DATA)
    candidates = [name]
    if not name.endswith((".yaml", ".yml")):
        candidates.extend([f"{name}.yaml", f"{name}.yml"])
    for candidate in candidates:
        path = data_dir.joinpath(candidate)
        if path.is_file():
            return Path(str(path)).resolve()
    choices = ", ".join(available_configs())
    raise ValueError(f"unknown packaged config {name!r}; choose one of: {choices}")


def load(config: str | Path | None = None) -> MarioRLConfig:
    """Load a packaged config name or YAML file path into a typed object."""
    if config is None:
        return MarioRLConfig()
    path = _resolve_config(config)
    data = _load_yaml_mapping(path)
    return from_mapping(data)


def from_mapping(data: Mapping[str, Any]) -> MarioRLConfig:
    """Build :class:`MarioRLConfig` from a nested mapping."""
    known = _TOP_LEVEL_FIELDS | set(_SECTIONS)
    unknown = set(data) - known
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown config field(s): {names}")

    values: dict[str, Any] = {
        "experiment_name": data.get("experiment_name", MarioRLConfig.experiment_name),
        "save_dir": data.get("save_dir", MarioRLConfig.save_dir),
    }
    for section_name, section_type in _SECTIONS.items():
        raw_section = data.get(section_name, {})
        if raw_section is None:
            raw_section = {}
        if not isinstance(raw_section, Mapping):
            raise TypeError(f"{section_name!r} must be a mapping")
        values[section_name] = _section_from_mapping(section_type, raw_section)
    return MarioRLConfig(**values)


def parse_cli_config(argv: Sequence[str] | None = None, *, description: str | None = None) -> MarioRLConfig:
    """Parse ``--config`` plus nested ``--section.field value`` overrides."""
    parser = build_parser(description=description)
    namespace = parser.parse_args(argv)
    values = _namespace_values(namespace)
    config_name = values.get("config")
    cfg = load(config_name) if config_name else MarioRLConfig()

    overrides: dict[str, Any] = {}
    for path in _override_paths():
        value = _value_for_path(values, path)
        if value is not _UNSET:
            overrides[path] = value
    return apply_overrides(cfg, overrides)


def cli(
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
    runner=None,
):
    """Reusable entrypoint helper for config-driven commands."""
    cfg = parse_cli_config(argv, description=description)
    if runner is None:
        return cfg
    return runner(cfg)


def build_parser(*, description: str | None = None):
    """Create the generated config parser used by command modules."""
    parser = _ArgumentParser(description=description or "Mario RL command")
    parser.add_argument(
        "--config",
        metavar="NAME_OR_PATH",
        default=None,
        help="packaged config name or YAML config file path",
    )
    parser.add_argument(
        "--experiment_name",
        dest="experiment_name",
        metavar="VALUE",
        default=argparse.SUPPRESS,
        help=f"experiment name (default: {MarioRLConfig.experiment_name})",
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        metavar="VALUE",
        default=argparse.SUPPRESS,
        help=f"artifact root directory (default: {MarioRLConfig.save_dir})",
    )
    for section_name, section_type in _SECTIONS.items():
        section = section_type()
        for field in fields(section):
            path = f"{section_name}.{field.name}"
            parser.add_argument(
                f"--{path}",
                dest=path,
                metavar="VALUE",
                default=argparse.SUPPRESS,
                help=f"{path} (default: {getattr(section, field.name)!r})",
            )
    return parser


def apply_overrides(config: MarioRLConfig, overrides: Mapping[str, Any]) -> MarioRLConfig:
    """Return a config with string CLI overrides applied and coerced."""
    result = config
    for path, raw_value in overrides.items():
        if path in _TOP_LEVEL_FIELDS:
            result = replace(result, **{path: str(raw_value)})
            continue
        try:
            section_name, field_name = path.split(".", 1)
        except ValueError as exc:
            raise ValueError(f"invalid override path {path!r}") from exc
        if section_name not in _SECTIONS:
            raise ValueError(f"unknown override section {section_name!r}")
        section = getattr(result, section_name)
        if not hasattr(section, field_name):
            raise ValueError(f"unknown override field {path!r}")
        current = getattr(section, field_name)
        if path == "model.num_actions":
            value = _coerce_num_actions_config_value(raw_value)
        else:
            value = _coerce_cli_value(raw_value, current)
        result = replace(result, **{section_name: replace(section, **{field_name: value})})
    return result


def to_dict(config: MarioRLConfig) -> dict[str, Any]:
    """Return a JSON/YAML-friendly dictionary for a typed config."""
    return asdict(config)


def action_space_summary(config: MarioRLConfig, *, env=None) -> dict[str, int | str | bool]:
    """Return resolved action-space metadata for a config or constructed env."""
    from mario_rl.envs.actions import action_set_summary as _action_set_summary

    return _action_set_summary(config.env.action_set, env=env)


def resolve_model_num_actions(config: MarioRLConfig, *, env=None) -> int:
    """Return a concrete model action count, validating fixed config values."""
    summary = action_space_summary(config, env=env)
    action_count = int(summary["action_count"])
    requested = getattr(config.model, "num_actions", AUTO_NUM_ACTIONS)
    if _is_auto_num_actions(requested):
        return action_count

    fixed = _coerce_num_actions_config_value(requested)
    if fixed != action_count:
        action_set = summary["action_set"]
        raise ValueError(
            f"model.num_actions={fixed} does not match env.action_set "
            f"{action_set!r} ({action_count} actions)"
        )
    return fixed


def with_resolved_model_num_actions(config: MarioRLConfig, *, env=None) -> MarioRLConfig:
    """Return ``config`` with automatic ``model.num_actions`` resolved to an int."""
    num_actions = resolve_model_num_actions(config, env=env)
    if config.model.num_actions == num_actions:
        return config
    return replace(config, model=replace(config.model, num_actions=num_actions))


def _override_paths() -> tuple[str, ...]:
    paths = ["experiment_name", "save_dir"]
    for section_name, section_type in _SECTIONS.items():
        paths.extend(f"{section_name}.{field.name}" for field in fields(section_type()))
    return tuple(paths)


def _resolve_config(config: str | Path) -> Path:
    path = Path(config).expanduser()
    if path.is_file():
        return path.resolve()
    return config_path(str(config))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        data = _load_simple_yaml(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must contain a YAML mapping")
    return dict(data)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small mapping-only YAML subset used by packaged configs."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            raise ValueError(f"invalid config line: {raw_line!r}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return json.loads(value)
    if value.startswith("{") and value.endswith("}"):
        return json.loads(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _section_from_mapping(section_type, data: Mapping[str, Any]):
    defaults = section_type()
    known = {field.name for field in fields(defaults)}
    unknown = set(data) - known
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {section_type.__name__} field(s): {names}")
    values = {}
    for field in fields(defaults):
        if field.name in data:
            if section_type is ModelConfig and field.name == "num_actions":
                values[field.name] = _coerce_num_actions_config_value(data[field.name])
            else:
                values[field.name] = _coerce_loaded_value(
                    data[field.name],
                    getattr(defaults, field.name),
                )
    return section_type(**values)


def _coerce_loaded_value(value: Any, default: Any) -> Any:
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _coerce_cli_value(value: Any, current: Any) -> Any:
    if not isinstance(value, str):
        return _coerce_loaded_value(value, current)
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if isinstance(current, bool):
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"expected boolean value, got {value!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        if value.startswith("["):
            parsed = json.loads(value)
        else:
            parsed = [part.strip() for part in value.split(",")]
        return tuple(_coerce_sequence_items(parsed, current))
    if isinstance(current, Mapping):
        if value.startswith("{"):
            parsed = json.loads(value)
            if not isinstance(parsed, Mapping):
                raise ValueError(f"expected mapping value, got {value!r}")
            return dict(parsed)
        pairs = [part.strip() for part in value.split(",") if part.strip()]
        result = {}
        for pair in pairs:
            key, separator, item_value = pair.partition("=")
            if not separator:
                raise ValueError(
                    f"expected comma-separated KEY=VALUE mapping, got {value!r}"
                )
            result[key.strip()] = _parse_scalar(item_value.strip())
        return result
    return value


def _coerce_sequence_items(values: Sequence[Any], current: tuple[Any, ...]) -> list[Any]:
    if not current:
        return list(values)
    exemplar = current[0]
    if isinstance(exemplar, int) and not isinstance(exemplar, bool):
        return [int(value) for value in values]
    if isinstance(exemplar, float):
        return [float(value) for value in values]
    return list(values)


def _is_auto_num_actions(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO_NUM_ACTIONS


def _coerce_num_actions_config_value(value: Any) -> int | str:
    if _is_auto_num_actions(value):
        return AUTO_NUM_ACTIONS
    if isinstance(value, bool):
        raise TypeError("model.num_actions must be a positive integer or 'auto'")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("model.num_actions must be > 0")
        return value
    if isinstance(value, str):
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("model.num_actions must be > 0")
        return parsed
    raise TypeError("model.num_actions must be a positive integer or 'auto'")


def _namespace_values(namespace) -> dict[str, Any]:
    if hasattr(namespace, "as_dict"):
        return namespace.as_dict()
    return vars(namespace)


def _value_for_path(values: Mapping[str, Any], path: str) -> Any:
    if path in values:
        return values[path]
    if "." not in path:
        return values.get(path, _UNSET)
    section_name, field_name = path.split(".", 1)
    section = values.get(section_name, _UNSET)
    if isinstance(section, Mapping):
        return section.get(field_name, _UNSET)
    if section is not _UNSET and hasattr(section, field_name):
        return getattr(section, field_name)
    return _UNSET


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for packaged config discovery."""
    parser = argparse.ArgumentParser(prog="python -m mario_rl.config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list packaged config names")
    path_parser = subparsers.add_parser("path", help="print a packaged config path")
    path_parser.add_argument("name")
    args = parser.parse_args(argv)

    if args.command == "list":
        for name in available_configs():
            print(name)
        return 0
    if args.command == "path":
        print(config_path(args.name))
        return 0
    raise AssertionError(f"unhandled config command {args.command!r}")


__all__ = [
    "AUTO_NUM_ACTIONS",
    "AuxiliaryLossConfig",
    "EnvConfig",
    "EpsilonConfig",
    "EvalConfig",
    "EvaluationMatrixConfig",
    "MarioRLConfig",
    "ModelConfig",
    "ReplayConfig",
    "RewardTransformConfig",
    "TaskSuiteConfig",
    "TrainConfig",
    "TrainerConfig",
    "apply_overrides",
    "action_space_summary",
    "available_configs",
    "build_parser",
    "cli",
    "config_path",
    "from_mapping",
    "load",
    "main",
    "parse_cli_config",
    "resolve_model_num_actions",
    "to_dict",
    "with_resolved_model_num_actions",
]
