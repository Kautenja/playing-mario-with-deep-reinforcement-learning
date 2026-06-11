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
from mario_rl.exploration import ExplorationConfig
from mario_rl.rewards import RewardTransformConfig

try:  # pragma: no cover - exercised when optional dependency is installed.
    from jsonargparse import ArgumentParser as _ArgumentParser
except ImportError:  # pragma: no cover - local test fallback.
    _ArgumentParser = argparse.ArgumentParser


AUTO_NUM_ACTIONS = "auto"
CUSTOM_PIXEL_PROFILE = "custom"


@dataclass(frozen=True)
class PixelObservationProfile:
    """Named preprocessing profile for pixel-only policy observations."""

    name: str
    description: str
    grayscale: bool
    image_size: tuple[int, int]
    frame_stack: int
    channel_first: bool = True
    interpolation: str = "area"

    @property
    def input_channels(self) -> int:
        """Return the channel count after grayscale/RGB conversion and stacking."""
        channels = 1 if self.grayscale else 3
        return channels * int(self.frame_stack)

    @property
    def state_shape(self) -> tuple[int, int, int]:
        """Return the channel-first observation shape produced by this profile."""
        height, width = self.image_size
        return (self.input_channels, height, width)


PIXEL_OBSERVATION_PROFILES: dict[str, PixelObservationProfile] = {
    "grayscale_84": PixelObservationProfile(
        name="grayscale_84",
        description="Fast grayscale 84x84 baseline for smoke tests and laptop gates.",
        grayscale=True,
        image_size=(84, 84),
        frame_stack=4,
    ),
    "rgb_balanced_90x96": PixelObservationProfile(
        name="rgb_balanced_90x96",
        description="Aspect-aware RGB profile for laptop smoke runs.",
        grayscale=False,
        image_size=(90, 96),
        frame_stack=4,
    ),
    "rgb_high_fidelity_120x128": PixelObservationProfile(
        name="rgb_high_fidelity_120x128",
        description="Aspect-aware RGB profile for longer experiments.",
        grayscale=False,
        image_size=(120, 128),
        frame_stack=4,
    ),
}

_PIXEL_PROFILE_FIELDS = (
    "image_size",
    "frame_stack",
    "grayscale",
    "channel_first",
    "interpolation",
)


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
    action_set: str = "complex"
    macro_actions: bool = False
    macro_action_set: str = "conservative"
    seed: int | None = 123
    pixel_profile: str = CUSTOM_PIXEL_PROFILE
    image_size: tuple[int, int] = (84, 84)
    frame_stack: int | None = 4
    reward_clipping: bool = False
    frame_skip: int | None = 4
    preprocess: bool = True
    grayscale: bool = True
    channel_first: bool = True
    interpolation: str = "area"
    record_statistics: bool = True
    max_episode_steps: int | None = 4000
    no_progress_timeout_steps: int | None = 600
    stuck_penalty: float = 0.01
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
            macro_actions=self.macro_actions,
            macro_action_set=self.macro_action_set,
            preprocess=self.preprocess,
            frame_skip=self.frame_skip,
            image_size=self.image_size,
            interpolation=self.interpolation,
            grayscale=self.grayscale,
            channel_first=self.channel_first,
            frame_stack=self.frame_stack,
            clip_rewards=self.reward_clipping,
            record_statistics=self.record_statistics,
            max_episode_steps=self.max_episode_steps,
            no_progress_timeout_steps=self.no_progress_timeout_steps,
            stuck_penalty=self.stuck_penalty,
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

    num_envs: int = 1
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
class ImitationConfig:
    """Local demonstration dataset and behavior-cloning pretraining settings."""

    data_dir: str = "data/imitation"
    validation_split: float = 0.2
    shuffle_seed: int = 123
    batch_size: int = 8
    max_epochs: int = 1
    max_steps: int = 16
    learning_rate: float | None = None
    checkpoint_name: str = "imitation-pretrain.ckpt"
    metrics_name: str = "imitation-metrics.json"
    num_workers: int = 0


@dataclass(frozen=True)
class SnapshotCurriculumConfig:
    """Process-local emulator snapshot curriculum settings."""

    enabled: bool = False
    max_snapshots: int = 32
    capture_interval_steps: int = 64
    sample_probability: float = 0.25
    min_progress: float | None = None
    rank_strategy: str = "progress"
    tags: tuple[str, ...] = ()


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
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    auxiliary: AuxiliaryLossConfig = field(default_factory=AuxiliaryLossConfig)
    evaluation_matrix: EvaluationMatrixConfig = field(default_factory=EvaluationMatrixConfig)
    replay: ReplayConfig = ReplayConfig()
    model: ModelConfig = ModelConfig()
    ppo: PPOConfig = PPOConfig()
    imitation: ImitationConfig = ImitationConfig()
    snapshot: SnapshotCurriculumConfig = SnapshotCurriculumConfig()
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
    "exploration": ExplorationConfig,
    "auxiliary": AuxiliaryLossConfig,
    "evaluation_matrix": EvaluationMatrixConfig,
    "replay": ReplayConfig,
    "model": ModelConfig,
    "ppo": PPOConfig,
    "imitation": ImitationConfig,
    "snapshot": SnapshotCurriculumConfig,
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
        return with_resolved_pixel_observation(
            MarioRLConfig(),
            allow_replay_state_shape_update=True,
            allow_model_input_channels_update=True,
        )
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
    raw_sections: dict[str, Mapping[str, Any]] = {}
    for section_name, section_type in _SECTIONS.items():
        raw_section = data.get(section_name, {})
        if raw_section is None:
            raw_section = {}
        if not isinstance(raw_section, Mapping):
            raise TypeError(f"{section_name!r} must be a mapping")
        raw_sections[section_name] = raw_section
        values[section_name] = _section_from_mapping(section_type, raw_section)
    config = MarioRLConfig(**values)
    config = _apply_pixel_profile(
        config,
        explicit_env_fields=set(raw_sections.get("env", ())),
    )
    return with_resolved_pixel_observation(
        config,
        allow_replay_state_shape_update="state_shape"
        not in raw_sections.get("replay", ()),
        allow_model_input_channels_update="input_channels"
        not in raw_sections.get("model", ()),
    )


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
    env_override_fields = {
        path.split(".", 1)[1]
        for path in overrides
        if path.startswith("env.")
    }
    if env_override_fields & set(_PIXEL_PROFILE_FIELDS) and "pixel_profile" not in env_override_fields:
        result = replace(
            result,
            env=replace(result.env, pixel_profile=CUSTOM_PIXEL_PROFILE),
        )
    result = _apply_pixel_profile(result, explicit_env_fields=env_override_fields)
    return with_resolved_pixel_observation(
        result,
        allow_replay_state_shape_update="replay.state_shape" not in overrides,
        allow_model_input_channels_update="model.input_channels" not in overrides,
    )


def to_dict(config: MarioRLConfig) -> dict[str, Any]:
    """Return a JSON/YAML-friendly dictionary for a typed config."""
    return asdict(config)


def pixel_observation_summary(config: MarioRLConfig) -> dict[str, Any]:
    """Return resolved pixel-observation metadata for artifacts."""
    resolved = with_resolved_pixel_observation(config)
    state_shape = tuple(int(dimension) for dimension in resolved.replay.state_shape)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a declared dependency.
        item_size = 1
    else:
        item_size = int(np.dtype(resolved.replay.sample_dtype).itemsize)
    return {
        "pixel_profile": resolved.env.pixel_profile,
        "grayscale": bool(resolved.env.grayscale),
        "image_size": list(resolved.env.image_size),
        "frame_stack": (
            int(resolved.env.frame_stack)
            if resolved.env.frame_stack is not None
            else 1
        ),
        "channel_first": bool(resolved.env.channel_first),
        "interpolation": resolved.env.interpolation,
        "state_shape": list(state_shape),
        "input_channels": int(resolved.model.input_channels),
        "sample_dtype": resolved.replay.sample_dtype,
        "bytes_per_observation": int(item_size * _product(state_shape)),
    }


def action_space_summary(config: MarioRLConfig, *, env=None) -> dict[str, object]:
    """Return resolved action-space metadata for a config or constructed env."""
    from mario_rl.envs.actions import action_set_summary as _action_set_summary

    return _action_set_summary(
        config.env.action_set,
        env=env,
        macro_actions=bool(config.env.macro_actions),
        macro_action_set=config.env.macro_action_set,
        frame_skip=config.env.frame_skip,
    )


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
    config = with_resolved_pixel_observation(config)
    num_actions = resolve_model_num_actions(config, env=env)
    if config.model.num_actions == num_actions:
        return config
    return replace(config, model=replace(config.model, num_actions=num_actions))


def with_resolved_pixel_observation(
    config: MarioRLConfig,
    *,
    allow_replay_state_shape_update: bool = False,
    allow_model_input_channels_update: bool = False,
) -> MarioRLConfig:
    """Return ``config`` with replay/model observation dimensions resolved."""
    config = _apply_pixel_profile(config, explicit_env_fields=set(_PIXEL_PROFILE_FIELDS))
    expected_shape = _expected_pixel_state_shape(config.env)
    expected_channels = expected_shape[0]
    result = config

    if tuple(int(dimension) for dimension in result.replay.state_shape) != expected_shape:
        if not allow_replay_state_shape_update:
            raise ValueError(
                "replay.state_shape "
                f"{tuple(result.replay.state_shape)} does not match resolved "
                f"pixel observation shape {expected_shape}"
            )
        result = replace(
            result,
            replay=replace(result.replay, state_shape=expected_shape),
        )

    if int(result.model.input_channels) != expected_channels:
        if not allow_model_input_channels_update:
            raise ValueError(
                f"model.input_channels={result.model.input_channels} does not "
                f"match resolved pixel channel count {expected_channels}"
            )
        result = replace(
            result,
            model=replace(result.model, input_channels=expected_channels),
        )
    return result


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


def _apply_pixel_profile(
    config: MarioRLConfig,
    *,
    explicit_env_fields: set[str],
) -> MarioRLConfig:
    profile_name = str(config.env.pixel_profile or CUSTOM_PIXEL_PROFILE)
    if profile_name == CUSTOM_PIXEL_PROFILE:
        return config
    profile = PIXEL_OBSERVATION_PROFILES.get(profile_name)
    if profile is None:
        choices = ", ".join(
            (CUSTOM_PIXEL_PROFILE, *sorted(PIXEL_OBSERVATION_PROFILES))
        )
        raise ValueError(
            f"unknown env.pixel_profile {profile_name!r}; choose one of: {choices}"
        )

    updates = {}
    for field_name in _PIXEL_PROFILE_FIELDS:
        expected = getattr(profile, field_name)
        current = getattr(config.env, field_name)
        if field_name in explicit_env_fields:
            if current != expected:
                raise ValueError(
                    f"env.{field_name}={current!r} does not match "
                    f"pixel profile {profile_name!r} expected {expected!r}; "
                    f"set env.pixel_profile={CUSTOM_PIXEL_PROFILE!r} for custom pixels"
                )
        else:
            updates[field_name] = expected
    if not updates:
        return config
    return replace(config, env=replace(config.env, **updates))


def _expected_pixel_state_shape(env: EnvConfig) -> tuple[int, int, int]:
    if not env.preprocess:
        raise ValueError("env.preprocess must be true for pixel-only training configs")
    if not env.channel_first:
        raise ValueError("env.channel_first must be true for policy/model observations")
    image_size = tuple(int(dimension) for dimension in env.image_size)
    if len(image_size) != 2 or any(dimension <= 0 for dimension in image_size):
        raise ValueError(f"env.image_size must be two positive integers, got {env.image_size!r}")
    frame_stack = 1 if env.frame_stack is None else int(env.frame_stack)
    if frame_stack <= 0:
        raise ValueError(f"env.frame_stack must be positive or null, got {env.frame_stack!r}")
    base_channels = 1 if env.grayscale else 3
    height, width = image_size
    return (base_channels * frame_stack, height, width)


def _product(values: Sequence[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


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
    "CUSTOM_PIXEL_PROFILE",
    "AuxiliaryLossConfig",
    "EnvConfig",
    "EpsilonConfig",
    "EvalConfig",
    "EvaluationMatrixConfig",
    "ExplorationConfig",
    "ImitationConfig",
    "MarioRLConfig",
    "ModelConfig",
    "PIXEL_OBSERVATION_PROFILES",
    "PixelObservationProfile",
    "ReplayConfig",
    "RewardTransformConfig",
    "TaskSuiteConfig",
    "TrainConfig",
    "TrainerConfig",
    "SnapshotCurriculumConfig",
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
    "pixel_observation_summary",
    "resolve_model_num_actions",
    "to_dict",
    "with_resolved_model_num_actions",
    "with_resolved_pixel_observation",
]
