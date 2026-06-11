"""Random rollout command artifact contract tests."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.random import run
from mario_rl.tests.fakes import fake_env_factory, tiny_ppo_config


class RandomCliTest(TestCase):
    """Validate random rollout payloads for configured action spaces."""

    def test_random_run_reports_macro_action_metadata(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                env=replace(
                    config.env,
                    macro_actions=True,
                    macro_action_set="conservative",
                    max_smoke_steps=3,
                ),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual("random", payload["command"])
            self.assertTrue(payload["macro_actions_enabled"])
            self.assertEqual("conservative", payload["macro_action_set"])
            self.assertEqual(payload["action_count"], payload["macro_action_count"])
            self.assertGreater(payload["action_count"], payload["base_action_count"])
            self.assertEqual(3, payload["steps"])
