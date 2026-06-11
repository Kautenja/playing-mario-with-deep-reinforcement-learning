"""Play/evaluation command artifact contract tests."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.play import run as play_run
from mario_rl.tests.fakes import fake_env_factory, tiny_ppo_config, tiny_training_config
from mario_rl.train import run as train_run


class PlayCliTest(TestCase):
    """Validate checkpoint evaluation writes metrics under the experiment dir."""

    def test_play_loads_smoke_checkpoint_and_writes_eval_metrics(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(0, train_run(config, env_factory=fake_env_factory))

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, play_run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual("play", payload["command"])
            self.assertEqual("complex", payload["action_set"])
            self.assertEqual(12, payload["action_count"])
            self.assertFalse(payload["native_action_space"])
            self.assertEqual(1, payload["episode_count"])
            self.assertLessEqual(payload["total_steps"], config.eval.max_steps)
            self.assertTrue(Path(payload["checkpoint"]).is_file())
            metrics_path = Path(payload["metrics_path"])
            self.assertTrue(metrics_path.is_file())
            metrics = json.loads(metrics_path.read_text())
            self.assertEqual(1, metrics["episode_count"])
            self.assertEqual("complex", metrics["action_set"])
            self.assertEqual(12, metrics["action_count"])
            self.assertIn("global", metrics)
            self.assertIn("by_game_family", metrics)
            self.assertIn("by_task", metrics)
            self.assertEqual(1, metrics["global"]["episode_count"])
            self.assertEqual(metrics["total_steps"], metrics["global"]["step_count"])
            self.assertIn("transformed_return", metrics["episodes"][0])

    def test_play_loads_vectorized_ppo_checkpoint_single_policy(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                ppo=replace(config.ppo, num_envs=2),
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(0, train_run(config, env_factory=fake_env_factory))

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, play_run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual("play", payload["command"])
            self.assertEqual("ppo", payload["algorithm"])
            self.assertEqual(1, payload["episode_count"])
            self.assertLessEqual(payload["total_steps"], config.eval.max_steps)
            self.assertTrue(Path(payload["checkpoint"]).is_file())

    def test_play_loads_macro_action_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                experiment_name="fake_ppo_macro_eval",
                env=replace(
                    config.env,
                    macro_actions=True,
                    macro_action_set="conservative",
                ),
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(0, train_run(config, env_factory=fake_env_factory))

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, play_run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual("play", payload["command"])
            self.assertEqual("ppo", payload["algorithm"])
            self.assertTrue(payload["macro_actions_enabled"])
            self.assertEqual("conservative", payload["macro_action_set"])
            self.assertEqual(payload["action_count"], payload["macro_action_count"])
            self.assertGreater(payload["action_count"], payload["base_action_count"])
            self.assertEqual(1, payload["episode_count"])
