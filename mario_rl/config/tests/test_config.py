"""Typed config and config CLI contract tests."""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

from mario_rl.config import (
    EnvConfig,
    EvalConfig,
    MarioRLConfig,
    ModelConfig,
    ReplayConfig,
    TaskSuiteConfig,
    TrainConfig,
    TrainerConfig,
    available_configs,
    config_path,
    load,
    main,
    parse_cli_config,
)


class ConfigSchemaTest(TestCase):
    """Validate the single typed config tree used by future specs."""

    def test_schema_covers_planned_training_surface(self):
        config = MarioRLConfig()

        self.assertIsInstance(config.trainer, TrainerConfig)
        self.assertIsInstance(config.task_suite, TaskSuiteConfig)
        self.assertIn(config.experiment_name, "smb_dqn_fast_dev")
        self.assertEqual("runs", config.save_dir)

        env_fields = EnvConfig.__dataclass_fields__
        for name in (
            "id",
            "render_mode",
            "action_set",
            "seed",
            "image_size",
            "frame_stack",
            "reward_clipping",
            "frame_skip",
            "video_enabled",
            "record_statistics",
            "max_smoke_steps",
        ):
            self.assertIn(name, env_fields)

        replay_fields = ReplayConfig.__dataclass_fields__
        for name in (
            "capacity",
            "batch_size",
            "warmup",
            "prioritized",
            "priority_alpha",
            "priority_beta",
            "sample_dtype",
            "state_shape",
        ):
            self.assertIn(name, replay_fields)

        model_fields = ModelConfig.__dataclass_fields__
        for name in (
            "architecture",
            "input_channels",
            "hidden_size",
            "optimizer",
            "learning_rate",
            "discount_factor",
            "double_dqn",
            "target_update_frequency",
            "compile",
            "task_conditioning",
            "task_feature_size",
        ):
            self.assertIn(name, model_fields)

        train_fields = TrainConfig.__dataclass_fields__
        for name in (
            "max_frames",
            "max_steps",
            "fast_dev_run",
            "log_interval",
            "checkpoint_path",
            "checkpoint_name",
            "metrics_name",
        ):
            self.assertIn(name, train_fields)

        self.assertIn("checkpoint", EvalConfig.__dataclass_fields__)

        task_suite_fields = TaskSuiteConfig.__dataclass_fields__
        for name in (
            "enabled",
            "game_families",
            "single_stage",
            "splits",
            "include_validated",
            "include_aliases",
            "family_weights",
            "seed",
            "switch_interval_episodes",
        ):
            self.assertIn(name, task_suite_fields)

    def test_packaged_configs_are_discoverable_and_load_typed_objects(self):
        names = available_configs()

        self.assertIn("smb_dqn_fast_dev", names)
        self.assertIn("smb_dqn_macbook_gate", names)
        self.assertIn("smb_dqn_task_conditioned_fast_dev", names)
        self.assertIn("smb_dqn_task_suite_fast_dev", names)
        self.assertIn("smb_dqn_cpu", names)
        self.assertIn("smb_dqn_mps", names)

        path = config_path("smb_dqn_fast_dev")
        self.assertTrue(path.is_absolute())
        self.assertTrue(path.is_file())

        config = load("smb_dqn_fast_dev")
        self.assertIsInstance(config, MarioRLConfig)
        self.assertEqual("SuperMarioBros-1-1-v0", config.env.id)
        self.assertEqual((84, 84), config.env.image_size)
        self.assertEqual((4, 84, 84), config.replay.state_shape)

        conditioned = load("smb_dqn_task_conditioned_fast_dev")
        self.assertTrue(conditioned.model.task_conditioning)
        self.assertEqual(0, conditioned.model.task_feature_size)

        task_suite = load("smb_dqn_task_suite_fast_dev")
        self.assertTrue(task_suite.task_suite.enabled)
        self.assertEqual(("smb1", "smb3"), task_suite.task_suite.game_families)
        self.assertEqual(
            {"smb1": 1.0, "smb3": 1.0},
            task_suite.task_suite.family_weights,
        )

        from_path = load(path)
        self.assertEqual(config, from_path)


class ConfigCliTest(TestCase):
    """Validate packaged config commands and nested overrides."""

    def test_config_cli_lists_and_prints_paths(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(0, main(["list"]))
        self.assertIn("smb_dqn_fast_dev", output.getvalue())

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(0, main(["path", "smb_dqn_fast_dev"]))
        self.assertTrue(Path(output.getvalue().strip()).is_file())

    def test_config_names_paths_and_nested_overrides_work(self):
        config = parse_cli_config(
            [
                "--config",
                "smb_dqn_fast_dev",
                "--train.fast_dev_run",
                "true",
                "--env.id",
                "SuperMarioBros-1-1-v0",
                "--env.image_size",
                "20,24",
                "--task_suite.enabled",
                "true",
                "--task_suite.family_weights",
                "smb1=1,smb3=2",
            ]
        )

        self.assertTrue(config.train.fast_dev_run)
        self.assertEqual("SuperMarioBros-1-1-v0", config.env.id)
        self.assertEqual((20, 24), config.env.image_size)
        self.assertTrue(config.task_suite.enabled)
        self.assertEqual({"smb1": 1, "smb3": 2}, config.task_suite.family_weights)

        with NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(
                "experiment_name: path_config\n"
                "env:\n"
                "  id: SuperMarioBros3-1-1-v0\n"
                "train:\n"
                "  fast_dev_run: true\n"
            )
            config_file.flush()

            loaded = parse_cli_config(["--config", config_file.name])
            self.assertEqual("path_config", loaded.experiment_name)
            self.assertEqual("SuperMarioBros3-1-1-v0", loaded.env.id)
            self.assertTrue(loaded.train.fast_dev_run)

    def test_bare_key_value_overrides_are_rejected(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_cli_config(["--config", "smb_dqn_fast_dev", "train.fast_dev_run=true"])
