"""Play/evaluation command artifact contract tests."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.play import run as play_run
from mario_rl.tests.fakes import fake_env_factory, tiny_training_config
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
            self.assertEqual("simple", payload["action_set"])
            self.assertEqual(7, payload["action_count"])
            self.assertFalse(payload["native_action_space"])
            self.assertEqual(1, payload["episode_count"])
            self.assertLessEqual(payload["total_steps"], config.eval.max_steps)
            self.assertTrue(Path(payload["checkpoint"]).is_file())
            metrics_path = Path(payload["metrics_path"])
            self.assertTrue(metrics_path.is_file())
            metrics = json.loads(metrics_path.read_text())
            self.assertEqual(1, metrics["episode_count"])
            self.assertEqual("simple", metrics["action_set"])
            self.assertEqual(7, metrics["action_count"])
