"""Training command artifact contract tests."""
from __future__ import annotations

import io
import csv
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.auxiliary import AuxiliaryLossConfig
from mario_rl.tests.fakes import fake_env_factory, tiny_ppo_config, tiny_training_config
from mario_rl.train import run


class TrainCliTest(TestCase):
    """Validate training writes checkpoint, config, and metrics artifacts."""

    def test_train_run_with_fake_env_writes_smoke_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual("train", payload["command"])
            self.assertEqual("simple", payload["action_set"])
            self.assertEqual(7, payload["action_count"])
            self.assertFalse(payload["native_action_space"])
            self.assertGreaterEqual(payload["global_step"], 1)
            self.assertEqual(config.train.max_steps, payload["env_frames"])
            self.assertTrue(Path(payload["checkpoint"]).is_file())
            self.assertTrue(Path(payload["metrics"]).is_file())
            self.assertTrue(Path(payload["metrics_json"]).is_file())
            with Path(payload["metrics"]).open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))[-1]
            self.assertEqual("simple", metrics["action_set"])
            self.assertEqual("7", metrics["action_count"])
            self.assertIn("metric_episode_count", metrics)
            self.assertIn("clear_rate", metrics)
            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())
            self.assertEqual("train", structured_metrics["command"])
            self.assertIn("global", structured_metrics)
            self.assertIn("by_task", structured_metrics)
            self.assertGreaterEqual(structured_metrics["global"]["step_count"], 1)
            self.assertTrue(Path(payload["resolved_config"]).is_file())
            tensorboard_dir = Path(payload["tensorboard"])
            self.assertTrue(tensorboard_dir.is_dir())
            self.assertTrue(
                list(tensorboard_dir.glob("events.out.tfevents.*")),
                list(tensorboard_dir.iterdir()),
            )
            self.assertIn("fake_lightning", Path(payload["resolved_config"]).read_text())

    def test_train_run_selects_ppo_and_writes_smoke_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                auxiliary=AuxiliaryLossConfig(
                    enabled=True,
                    targets=("progress_delta", "clear", "game_family"),
                    weights={"game_family": 0.5},
                    head_hidden_size=16,
                ),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual("train", payload["command"])
            self.assertEqual("ppo", payload["algorithm"])
            self.assertEqual(config.train.max_steps * config.ppo.rollout_steps, payload["env_frames"])
            self.assertTrue(Path(payload["checkpoint"]).is_file())
            self.assertTrue(Path(payload["metrics_json"]).is_file())
            with Path(payload["metrics"]).open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))[-1]
            self.assertIn("ppo_policy_loss", metrics)
            self.assertIn("auxiliary_loss", metrics)
            self.assertIn("game_family", metrics["auxiliary_losses_json"])
            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())
            self.assertEqual("ppo", structured_metrics["algorithm"])
            self.assertIn("ppo", structured_metrics)
            self.assertIn("auxiliary", structured_metrics)
            self.assertTrue(structured_metrics["auxiliary"]["enabled"])
            self.assertIn("game_family", structured_metrics["auxiliary"]["losses"])
            self.assertIn("global", structured_metrics)
            self.assertIn("fake_ppo_lightning", Path(payload["resolved_config"]).read_text())
