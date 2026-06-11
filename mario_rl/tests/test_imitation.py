"""Imitation dataset and behavior-cloning command tests."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from mario_rl.config import action_space_summary, with_resolved_model_num_actions
from mario_rl.imitation import (
    load_imitation_dataset,
    run as run_pretrain,
    split_imitation_dataset,
)
from mario_rl.tests.fakes import fake_env_factory, tiny_ppo_config
from mario_rl.train import run as run_train


class ImitationDatasetTest(TestCase):
    """Validate local pixel demonstration loading and pretraining."""

    def test_loader_splits_deterministically_and_reports_histogram(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            data_dir = Path(tmpdir) / "demos"
            _write_synthetic_dataset(data_dir, config, steps=8)

            dataset = load_imitation_dataset(config, data_dir=data_dir)
            train_a, validation_a = split_imitation_dataset(
                dataset,
                validation_split=0.25,
                seed=7,
            )
            train_b, validation_b = split_imitation_dataset(
                dataset,
                validation_split=0.25,
                seed=7,
            )

            self.assertEqual(8, len(dataset))
            self.assertEqual(6, len(train_a))
            self.assertEqual(2, len(validation_a))
            np.testing.assert_array_equal(train_a.actions, train_b.actions)
            np.testing.assert_array_equal(validation_a.actions, validation_b.actions)
            self.assertEqual(8, sum(dataset.action_histogram(12)))
            self.assertEqual(("FakeMario-v0",), dataset.env_ids)

    def test_validation_catches_action_and_pixel_contract_mismatches(self):
        cases = (
            (
                "action count",
                {},
                {"action_count": 11},
            ),
            (
                "channel count",
                {"observation_shape": (3, 84, 84)},
                {},
            ),
            (
                "image size",
                {"observation_shape": (4, 80, 84)},
                {},
            ),
            (
                "frame stack",
                {},
                {"frame_stack": 2},
            ),
        )
        for message, dataset_kwargs, metadata_updates in cases:
            with self.subTest(message=message), TemporaryDirectory() as tmpdir:
                config = tiny_ppo_config(tmpdir)
                data_dir = Path(tmpdir) / "demos"
                _write_synthetic_dataset(
                    data_dir,
                    config,
                    metadata_updates=metadata_updates,
                    **dataset_kwargs,
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_imitation_dataset(config, data_dir=data_dir)

    def test_pretrain_writes_checkpoint_and_train_can_load_it(self):
        with TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "demos"
            config = tiny_ppo_config(tmpdir)
            _write_synthetic_dataset(data_dir, config, steps=8)
            pretrain_config = replace(
                config,
                experiment_name="fake_imitation_pretrain",
                imitation=replace(
                    config.imitation,
                    data_dir=str(data_dir),
                    validation_split=0.25,
                    batch_size=2,
                    max_epochs=1,
                    max_steps=2,
                    checkpoint_name="pretrain.ckpt",
                    metrics_name="pretrain-metrics.json",
                ),
                trainer=replace(config.trainer, enable_progress_bar=False),
            )

            pretrain_output = io.StringIO()
            with redirect_stdout(pretrain_output):
                self.assertEqual(0, run_pretrain(pretrain_config))
            pretrain_payload = json.loads(pretrain_output.getvalue().splitlines()[-1])

            checkpoint = Path(pretrain_payload["checkpoint"])
            metrics = Path(pretrain_payload["metrics"])
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(metrics.is_file())
            self.assertEqual("pretrain", pretrain_payload["command"])
            self.assertEqual(8, pretrain_payload["dataset"]["dataset_size"])
            self.assertEqual(6, pretrain_payload["dataset"]["train_size"])
            self.assertEqual(2, pretrain_payload["dataset"]["validation_size"])
            self.assertEqual(8, sum(pretrain_payload["dataset"]["action_histogram"]))
            self.assertGreater(pretrain_payload["cross_entropy_loss"], 0.0)
            self.assertIsNotNone(pretrain_payload["validation_accuracy"])

            train_config = replace(
                tiny_ppo_config(tmpdir),
                experiment_name="fake_ppo_after_imitation",
                train=replace(
                    config.train,
                    checkpoint_path=str(checkpoint),
                    max_steps=1,
                    checkpoint_name="after-pretrain.ckpt",
                ),
                trainer=replace(config.trainer, enable_progress_bar=False),
            )
            train_output = io.StringIO()
            with redirect_stdout(train_output):
                self.assertEqual(0, run_train(train_config, env_factory=fake_env_factory))
            train_payload = json.loads(train_output.getvalue().splitlines()[-1])

            self.assertEqual("train", train_payload["command"])
            self.assertEqual("ppo", train_payload["algorithm"])
            self.assertTrue(Path(train_payload["checkpoint"]).is_file())


def _write_synthetic_dataset(
    data_dir: Path,
    config,
    *,
    steps: int = 6,
    observation_shape: tuple[int, int, int] | None = None,
    metadata_updates: dict[str, object] | None = None,
) -> Path:
    config = with_resolved_model_num_actions(config)
    action_summary = action_space_summary(config)
    action_count = int(action_summary["action_count"])
    shape = observation_shape or tuple(int(value) for value in config.replay.state_shape)
    values = np.arange(int(steps) * int(np.prod(shape)), dtype=np.uint32)
    observations = (values.reshape(int(steps), *shape) % 256).astype(np.uint8)
    actions = (np.arange(int(steps), dtype=np.int64) % action_count).astype(np.int64)
    terminated = np.zeros(int(steps), dtype=np.bool_)
    truncated = np.zeros(int(steps), dtype=np.bool_)
    terminated[-1] = True
    metadata = {
        "env_id": config.env.id,
        "action_set": action_summary["action_set"],
        "action_count": action_count,
        "macro_actions": bool(action_summary["macro_actions_enabled"]),
        "macro_action_set": action_summary.get("macro_action_set")
        or config.env.macro_action_set,
        "pixel_profile": config.env.pixel_profile,
        "observation_shape": list(shape),
        "image_size": [int(shape[1]), int(shape[2])],
        "frame_stack": int(config.env.frame_stack or 1),
        "channel_first": True,
        "source_notes": "synthetic unittest fixture",
    }
    metadata.update(metadata_updates or {})

    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tiny-demo.npz"
    np.savez(
        path,
        observations=observations,
        actions=actions,
        terminated=terminated,
        truncated=truncated,
        metadata=json.dumps(metadata),
    )
    return path
