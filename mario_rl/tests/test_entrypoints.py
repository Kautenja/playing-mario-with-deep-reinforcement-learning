"""Import-safe module entrypoint tests."""
from __future__ import annotations

import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from unittest import TestCase


class EntrypointTest(TestCase):
    """Validate import-safe config-driven entrypoints."""

    def test_entrypoint_imports_do_not_start_legacy_frameworks_or_work(self):
        for module_name in ("mario_rl.train", "mario_rl.play", "mario_rl.random"):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

        self.assertNotIn("tensorflow", sys.modules)
        self.assertNotIn("keras", sys.modules)

    def test_help_is_available_without_creating_environments(self):
        before_modules = set(sys.modules)
        for module_name in ("mario_rl.train", "mario_rl.play", "mario_rl.random"):
            module = importlib.import_module(module_name)
            output = io.StringIO()
            with self.subTest(module_name=module_name), redirect_stdout(output):
                with self.assertRaises(SystemExit) as context:
                    module.main(["--help"])
            self.assertEqual(0, context.exception.code)
            self.assertIn("--config", output.getvalue())
            self.assertIn("--train.fast_dev_run", output.getvalue())

        imported_by_help = set(sys.modules) - before_modules
        self.assertFalse(
            {"gym_super_mario_bros", "mario_rl.envs"} & imported_by_help,
            imported_by_help,
        )

    def test_train_and_play_parse_configs_without_training(self):
        for module_name, command in (
            ("mario_rl.train", "train"),
            ("mario_rl.play", "play"),
        ):
            module = importlib.import_module(module_name)
            output = io.StringIO()
            with self.subTest(module_name=module_name), redirect_stdout(output):
                self.assertEqual(
                    0,
                    module.main(
                        [
                            "--config",
                            "smb_dqn_fast_dev",
                            "--train.fast_dev_run",
                            "true",
                            "--env.id",
                            "SuperMarioBros1-1-v0",
                        ]
                    ),
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(command, payload["command"])
            self.assertEqual("SuperMarioBros1-1-v0", payload["config"]["env"]["id"])
            self.assertTrue(payload["config"]["train"]["fast_dev_run"])
