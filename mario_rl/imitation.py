"""Pixel-only imitation dataset loading and behavior-cloning pretraining."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mario_rl.config import (
    MarioRLConfig,
    action_space_summary,
    cli,
    pixel_observation_summary,
    with_resolved_model_num_actions,
)


IMITATION_CHECKPOINT_TYPE = "mario_rl.imitation_pretrain.v1"
FORBIDDEN_POLICY_INPUT_KEYS = {
    "ram",
    "info",
    "reward",
    "rewards",
    "reward_components",
    "task_features",
    "task_metadata",
    "object_map",
    "object_maps",
    "tile_map",
    "tile_maps",
}


@dataclass(frozen=True)
class ImitationDataset:
    """Flattened pixel observations and action labels from local demonstrations."""

    observations: np.ndarray
    actions: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_boundaries: np.ndarray
    files: tuple[Path, ...]
    metadata: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        """Return the number of labeled pixel observations."""
        return int(self.actions.shape[0])

    def subset(self, indices: Sequence[int]) -> "ImitationDataset":
        """Return a deterministic subset view copied into compact arrays."""
        index_array = np.asarray(indices, dtype=np.int64)
        return ImitationDataset(
            observations=self.observations[index_array],
            actions=self.actions[index_array],
            terminated=self.terminated[index_array],
            truncated=self.truncated[index_array],
            episode_boundaries=self.episode_boundaries[index_array],
            files=self.files,
            metadata=self.metadata,
        )

    def action_histogram(self, action_count: int) -> list[int]:
        """Return a dense action histogram with one entry per valid action."""
        histogram = np.bincount(
            self.actions.astype(np.int64),
            minlength=int(action_count),
        )
        return [int(value) for value in histogram[: int(action_count)]]

    @property
    def env_ids(self) -> tuple[str, ...]:
        """Return unique environment IDs declared by the loaded segments."""
        return tuple(sorted({str(item["env_id"]) for item in self.metadata}))


class _TorchDatasetAdapter:
    """Minimal map-style adapter consumed by ``torch.utils.data.DataLoader``."""

    def __init__(self, dataset: ImitationDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.int64]:
        return {
            "observation": self.dataset.observations[index],
            "action": np.int64(self.dataset.actions[index]),
        }


def load_imitation_dataset(
    config: MarioRLConfig,
    *,
    data_dir: str | Path | None = None,
) -> ImitationDataset:
    """Load and validate every ``*.npz`` imitation segment in ``data_dir``."""
    config = with_resolved_model_num_actions(config)
    root = Path(data_dir or config.imitation.data_dir).expanduser()
    files = tuple(sorted(root.glob("*.npz")))
    if not files:
        raise FileNotFoundError(
            f"no imitation .npz files found in {root}; place local demos there "
            "or override imitation.data_dir"
        )

    segments = [_load_segment(path, config) for path in files]
    observations = np.concatenate([segment.observations for segment in segments], axis=0)
    actions = np.concatenate([segment.actions for segment in segments], axis=0)
    terminated = np.concatenate([segment.terminated for segment in segments], axis=0)
    truncated = np.concatenate([segment.truncated for segment in segments], axis=0)
    boundaries = np.concatenate([segment.episode_boundaries for segment in segments], axis=0)
    metadata = tuple(segment.metadata for segment in segments)
    return ImitationDataset(
        observations=observations,
        actions=actions,
        terminated=terminated,
        truncated=truncated,
        episode_boundaries=boundaries,
        files=files,
        metadata=metadata,
    )


def split_imitation_dataset(
    dataset: ImitationDataset,
    *,
    validation_split: float,
    seed: int,
) -> tuple[ImitationDataset, ImitationDataset]:
    """Return deterministic train/validation splits."""
    total = len(dataset)
    if total <= 0:
        raise ValueError("imitation dataset is empty")
    fraction = float(validation_split)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("imitation.validation_split must be in [0.0, 1.0)")
    indices = np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    if fraction == 0.0 or total == 1:
        return dataset.subset(indices), dataset.subset([])
    validation_count = int(round(total * fraction))
    validation_count = max(1, min(validation_count, total - 1))
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return dataset.subset(train_indices), dataset.subset(validation_indices)


def run(config: MarioRLConfig, *, data_dir: str | Path | None = None) -> int:
    """Run behavior-cloning pretraining for a recurrent actor-critic policy."""
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    from torch.utils.data import DataLoader

    from mario_rl.lightning import (
        experiment_paths,
        trainer_accelerator,
        trainer_devices,
        write_json,
        write_resolved_config,
    )

    config = with_resolved_model_num_actions(config)
    if config.trainer.seed is not None:
        seed_everything(config.trainer.seed, workers=True)

    dataset = load_imitation_dataset(config, data_dir=data_dir)
    train_set, validation_set = split_imitation_dataset(
        dataset,
        validation_split=config.imitation.validation_split,
        seed=config.imitation.shuffle_seed,
    )
    if len(train_set) <= 0:
        raise ValueError("imitation training split is empty")

    paths = experiment_paths(config)
    write_resolved_config(config, paths.resolved_config)
    action_summary = action_space_summary(config)
    action_count = int(action_summary["action_count"])
    checkpoint_path = paths.checkpoints / config.imitation.checkpoint_name
    metrics_path = paths.root / config.imitation.metrics_name

    module = _make_behavior_cloning_module(config)
    train_loader = DataLoader(
        _TorchDatasetAdapter(train_set),
        batch_size=max(1, int(config.imitation.batch_size)),
        shuffle=False,
        num_workers=max(0, int(config.imitation.num_workers)),
    )
    validation_loader = (
        DataLoader(
            _TorchDatasetAdapter(validation_set),
            batch_size=max(1, int(config.imitation.batch_size)),
            shuffle=False,
            num_workers=max(0, int(config.imitation.num_workers)),
        )
        if len(validation_set) > 0
        else None
    )
    csv_logger = CSVLogger(save_dir=str(paths.logs), name="imitation-lightning")
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(paths.logs),
        name="imitation-tensorboard",
    )
    trainer = Trainer(
        accelerator=trainer_accelerator(config),
        devices=trainer_devices(config),
        precision=config.trainer.precision,
        deterministic=config.trainer.deterministic,
        default_root_dir=str(paths.root),
        max_epochs=max(1, int(config.imitation.max_epochs)),
        max_steps=max(1, int(config.imitation.max_steps)),
        logger=[csv_logger, tensorboard_logger],
        enable_checkpointing=False,
        enable_progress_bar=bool(config.trainer.enable_progress_bar),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )
    validation_accuracy = _evaluate_validation_accuracy(module, validation_loader)
    trainer.save_checkpoint(str(checkpoint_path))
    _mark_imitation_checkpoint(checkpoint_path, config=config)

    metrics = {
        "command": "pretrain",
        "checkpoint_type": IMITATION_CHECKPOINT_TYPE,
        "algorithm": "ppo",
        **action_summary,
        "pixel_observation": pixel_observation_summary(config),
        "dataset": {
            "data_dir": str(Path(data_dir or config.imitation.data_dir).expanduser()),
            "files": [str(path) for path in dataset.files],
            "env_ids": list(dataset.env_ids),
            "dataset_size": len(dataset),
            "train_size": len(train_set),
            "validation_size": len(validation_set),
            "action_histogram": dataset.action_histogram(action_count),
        },
        "cross_entropy_loss": module.training_loss_mean(),
        "validation_accuracy": validation_accuracy,
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "resolved_config": str(paths.resolved_config),
        "tensorboard": str(tensorboard_logger.log_dir),
    }
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, sort_keys=True))
    return 0


def is_imitation_checkpoint(path: str | Path) -> bool:
    """Return whether ``path`` points to a Mario imitation pretrain checkpoint."""
    checkpoint = _load_torch_checkpoint(path)
    return checkpoint.get("mario_rl_checkpoint_type") == IMITATION_CHECKPOINT_TYPE


def load_imitation_policy_weights(module: Any, path: str | Path) -> None:
    """Load policy weights from an imitation checkpoint into an RL module."""
    checkpoint = _load_torch_checkpoint(path)
    checkpoint_type = checkpoint.get("mario_rl_checkpoint_type")
    if checkpoint_type != IMITATION_CHECKPOINT_TYPE:
        raise ValueError(f"{path} is not an imitation pretrain checkpoint")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{path} does not contain a Lightning state_dict")
    module.load_state_dict(state_dict, strict=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse imitation config and execute behavior-cloning pretraining."""
    return cli(
        argv,
        description="Pretrain a recurrent Mario policy from local pixel demonstrations.",
        runner=run,
    )


@dataclass(frozen=True)
class _Segment:
    observations: np.ndarray
    actions: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_boundaries: np.ndarray
    metadata: dict[str, Any]


def _load_segment(path: Path, config: MarioRLConfig) -> _Segment:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        forbidden = keys & FORBIDDEN_POLICY_INPUT_KEYS
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"{path} contains forbidden policy input field(s): {names}")
        _require_keys(path, keys, ("observations", "actions"))
        metadata = _load_metadata(path, data)
        observations = np.asarray(data["observations"])
        actions = np.asarray(data["actions"], dtype=np.int64)
        _validate_arrays(path, observations, actions)
        terminated, truncated, boundaries = _load_boundaries(path, data, len(actions))
        _validate_segment_metadata(path, metadata, observations, actions, config)
        return _Segment(
            observations=observations.astype(np.uint8, copy=False),
            actions=actions,
            terminated=terminated,
            truncated=truncated,
            episode_boundaries=boundaries,
            metadata=metadata,
        )


def _make_behavior_cloning_module(config: MarioRLConfig):
    import torch
    import torch.nn.functional as functional
    from lightning.pytorch import LightningModule

    from mario_rl.models import RecurrentActorCritic, build_model

    class BehaviorCloningModule(LightningModule):
        """Lightning module that trains only on pixel observations and actions."""

        def __init__(self, cfg: MarioRLConfig) -> None:
            super().__init__()
            self.config = cfg
            self.save_hyperparameters({"config": _config_to_dict(cfg)})
            policy = build_model(cfg)
            if not isinstance(policy, RecurrentActorCritic):
                raise TypeError(
                    "imitation pretraining requires "
                    "model.architecture='recurrent_actor_critic'"
                )
            self.policy = policy
            self._train_loss_total = 0.0
            self._train_example_count = 0
            self._validation_correct = 0
            self._validation_total = 0

        def configure_optimizers(self):
            learning_rate = (
                float(self.config.imitation.learning_rate)
                if self.config.imitation.learning_rate is not None
                else float(self.config.model.learning_rate)
            )
            return torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        def training_step(self, batch, _batch_idx):
            loss, correct, total = self._step(batch)
            self._train_loss_total += float(loss.detach().cpu().item()) * total
            self._train_example_count += int(total)
            self.log("pretrain/cross_entropy", loss, on_step=True, prog_bar=False)
            self.log(
                "pretrain/train_accuracy",
                float(correct) / max(total, 1),
                on_step=True,
                prog_bar=False,
            )
            return loss

        def on_validation_epoch_start(self) -> None:
            self._validation_correct = 0
            self._validation_total = 0

        def validation_step(self, batch, _batch_idx):
            loss, correct, total = self._step(batch)
            self._validation_correct += int(correct)
            self._validation_total += int(total)
            self.log("pretrain/validation_cross_entropy", loss, on_epoch=True)
            self.log(
                "pretrain/validation_accuracy",
                float(correct) / max(total, 1),
                on_epoch=True,
            )
            return loss

        def training_loss_mean(self) -> float:
            if self._train_example_count <= 0:
                return 0.0
            return float(self._train_loss_total / self._train_example_count)

        def validation_accuracy(self) -> float | None:
            if self._validation_total <= 0:
                return None
            return float(self._validation_correct / self._validation_total)

        def _step(self, batch) -> tuple[torch.Tensor, int, int]:
            observations = batch["observation"].to(device=self.device)
            actions = batch["action"].to(device=self.device, dtype=torch.long)
            hidden_state = self.policy.initial_state(
                int(observations.shape[0]),
                device=self.device,
            )
            output = self.policy(observations, hidden_state, task_features=None)
            logits = output.policy_logits
            loss = functional.cross_entropy(logits, actions)
            predictions = torch.argmax(logits.detach(), dim=1)
            correct = int((predictions == actions).sum().detach().cpu().item())
            return loss, correct, int(actions.numel())

    return BehaviorCloningModule(config)


def _evaluate_validation_accuracy(module, validation_loader) -> float | None:
    if validation_loader is None:
        return None
    import torch

    correct = 0
    total = 0
    was_training = module.training
    module.eval()
    with torch.no_grad():
        for batch in validation_loader:
            _loss, batch_correct, batch_total = module._step(batch)
            correct += int(batch_correct)
            total += int(batch_total)
    if was_training:
        module.train()
    if total <= 0:
        return None
    return float(correct / total)


def _mark_imitation_checkpoint(path: Path, *, config: MarioRLConfig) -> None:
    checkpoint = _load_torch_checkpoint(path)
    checkpoint["mario_rl_checkpoint_type"] = IMITATION_CHECKPOINT_TYPE
    checkpoint["imitation"] = {
        "checkpoint_type": IMITATION_CHECKPOINT_TYPE,
        "data_dir": config.imitation.data_dir,
        "validation_split": float(config.imitation.validation_split),
        "shuffle_seed": int(config.imitation.shuffle_seed),
    }
    _save_torch_checkpoint(checkpoint, path)


def _load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        checkpoint = torch.load(Path(path).expanduser(), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a Lightning checkpoint dictionary")
    return checkpoint


def _save_torch_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    import torch

    torch.save(dict(checkpoint), Path(path).expanduser())


def _require_keys(path: Path, keys: set[str], required: Sequence[str]) -> None:
    missing = set(required) - keys
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required field(s): {names}")


def _load_metadata(path: Path, data) -> dict[str, Any]:
    if "metadata" in data.files:
        raw = data["metadata"]
        value = raw.reshape(-1)[0].item() if raw.size else ""
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        metadata = json.loads(str(value))
    else:
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"{path} requires a metadata field or {sidecar.name}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} metadata must be a JSON object")
    return metadata


def _load_boundaries(path: Path, data, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = set(data.files)
    has_terminal_flags = {"terminated", "truncated"} <= keys
    has_boundaries = "episode_boundaries" in keys
    if not has_terminal_flags and not has_boundaries:
        raise ValueError(
            f"{path} requires terminated/truncated flags or episode_boundaries"
        )
    if has_terminal_flags:
        terminated = _bool_vector(path, "terminated", data["terminated"], length)
        truncated = _bool_vector(path, "truncated", data["truncated"], length)
    else:
        terminated = np.zeros(length, dtype=np.bool_)
        truncated = np.zeros(length, dtype=np.bool_)
    if has_boundaries:
        boundaries = _bool_vector(path, "episode_boundaries", data["episode_boundaries"], length)
    else:
        boundaries = terminated | truncated
    return terminated, truncated, boundaries


def _bool_vector(path: Path, name: str, value: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    if array.shape != (length,):
        raise ValueError(f"{path} {name} must have shape ({length},), got {array.shape}")
    return array


def _validate_arrays(path: Path, observations: np.ndarray, actions: np.ndarray) -> None:
    if observations.ndim != 4:
        raise ValueError(
            f"{path} observations must have shape (steps, channels, height, width)"
        )
    if observations.dtype != np.uint8:
        raise ValueError(f"{path} observations must use uint8 pixels")
    if actions.ndim != 1:
        raise ValueError(f"{path} actions must have shape (steps,)")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            f"{path} observations/actions length mismatch: "
            f"{observations.shape[0]} != {actions.shape[0]}"
        )
    if actions.shape[0] <= 0:
        raise ValueError(f"{path} contains no action labels")


def _validate_segment_metadata(
    path: Path,
    metadata: Mapping[str, Any],
    observations: np.ndarray,
    actions: np.ndarray,
    config: MarioRLConfig,
) -> None:
    required = {
        "env_id",
        "action_set",
        "action_count",
        "macro_actions",
        "macro_action_set",
        "pixel_profile",
        "observation_shape",
        "image_size",
        "frame_stack",
        "channel_first",
    }
    missing = required - set(metadata)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path} metadata missing required field(s): {names}")

    action_summary = action_space_summary(config)
    expected_action_count = int(action_summary["action_count"])
    action_count = int(metadata["action_count"])
    if action_count != expected_action_count:
        raise ValueError(
            f"{path} action count {action_count} does not match target "
            f"{expected_action_count}"
        )
    if actions.min(initial=0) < 0 or actions.max(initial=0) >= expected_action_count:
        raise ValueError(f"{path} actions contain labels outside configured action space")
    expected_action_set = str(action_summary["action_set"])
    if str(metadata["action_set"]) != expected_action_set:
        raise ValueError(
            f"{path} action set {metadata['action_set']!r} does not match "
            f"target {expected_action_set!r}"
        )

    expected_macro = bool(action_summary["macro_actions_enabled"])
    if bool(metadata["macro_actions"]) != expected_macro:
        raise ValueError(
            f"{path} macro action setting does not match target config"
        )
    expected_macro_set = action_summary.get("macro_action_set") or config.env.macro_action_set
    if str(metadata["macro_action_set"]) != str(expected_macro_set):
        raise ValueError(
            f"{path} macro action set {metadata['macro_action_set']!r} does not "
            f"match target {expected_macro_set!r}"
        )

    if str(metadata["pixel_profile"]) != str(config.env.pixel_profile):
        raise ValueError(
            f"{path} pixel profile {metadata['pixel_profile']!r} does not match "
            f"target {config.env.pixel_profile!r}"
        )
    observation_shape = tuple(int(value) for value in observations.shape[1:])
    declared_shape = tuple(int(value) for value in metadata["observation_shape"])
    if declared_shape != observation_shape:
        raise ValueError(
            f"{path} metadata observation_shape {declared_shape} does not match "
            f"observations {observation_shape}"
        )
    _validate_observation_contract(path, metadata, observation_shape, config)


def _validate_observation_contract(
    path: Path,
    metadata: Mapping[str, Any],
    observation_shape: tuple[int, int, int],
    config: MarioRLConfig,
) -> None:
    expected_shape = tuple(int(value) for value in config.replay.state_shape)
    expected_image_size = tuple(int(value) for value in config.env.image_size)
    expected_frame_stack = 1 if config.env.frame_stack is None else int(config.env.frame_stack)
    expected_channels = int(config.model.input_channels)
    if int(metadata["frame_stack"]) != expected_frame_stack:
        raise ValueError(
            f"{path} frame stack {metadata['frame_stack']} does not match "
            f"target {expected_frame_stack}"
        )
    if bool(metadata["channel_first"]) is not True:
        raise ValueError(f"{path} demonstrations must be channel-first pixels")
    declared_image_size = tuple(int(value) for value in metadata["image_size"])
    if declared_image_size != expected_image_size:
        raise ValueError(
            f"{path} image size {declared_image_size} does not match "
            f"target {expected_image_size}"
        )
    if observation_shape[0] != expected_channels:
        raise ValueError(
            f"{path} channel count {observation_shape[0]} does not match "
            f"target {expected_channels}"
        )
    if observation_shape[1:] != expected_image_size:
        raise ValueError(
            f"{path} image size {observation_shape[1:]} does not match "
            f"target {expected_image_size}"
        )
    if observation_shape != expected_shape:
        raise ValueError(
            f"{path} observation shape {observation_shape} does not match "
            f"target {expected_shape}"
        )


def _config_to_dict(config: MarioRLConfig) -> dict[str, Any]:
    from mario_rl.config import to_dict

    return to_dict(config)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORBIDDEN_POLICY_INPUT_KEYS",
    "IMITATION_CHECKPOINT_TYPE",
    "ImitationDataset",
    "is_imitation_checkpoint",
    "load_imitation_dataset",
    "load_imitation_policy_weights",
    "main",
    "run",
    "split_imitation_dataset",
]
