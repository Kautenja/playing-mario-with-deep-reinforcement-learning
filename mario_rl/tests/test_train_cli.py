"""Training command artifact contract tests."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.tests.fakes import fake_env_factory, tiny_training_config
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
            self.assertGreaterEqual(payload["global_step"], 1)
            self.assertEqual(config.train.max_steps, payload["env_frames"])
            self.assertTrue(Path(payload["checkpoint"]).is_file())
            self.assertTrue(Path(payload["metrics"]).is_file())
            self.assertTrue(Path(payload["resolved_config"]).is_file())
            self.assertIn("fake_lightning", Path(payload["resolved_config"]).read_text())
