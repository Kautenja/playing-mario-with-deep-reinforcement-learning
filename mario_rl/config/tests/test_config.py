"""Typed config and config CLI contract tests."""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

from mario_rl.config import (
    AUTO_NUM_ACTIONS,
    EnvConfig,
    EvalConfig,
    MarioRLConfig,
    ModelConfig,
    PPOConfig,
    ReplayConfig,
    RewardTransformConfig,
    TaskSuiteConfig,
    TrainConfig,
    TrainerConfig,
    available_configs,
    config_path,
    load,
    main,
    parse_cli_config,
    resolve_model_num_actions,
    with_resolved_model_num_actions,
)


class ConfigSchemaTest(TestCase):
    """Validate the single typed config tree used by future specs."""

    def test_schema_covers_planned_training_surface(self):
        config = MarioRLConfig()

        self.assertIsInstance(config.trainer, TrainerConfig)
        self.assertIsInstance(config.task_suite, TaskSuiteConfig)
        self.assertIsInstance(config.reward_transform, RewardTransformConfig)
        self.assertIsInstance(config.ppo, PPOConfig)
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
            "store_reward_info",
        ):
            self.assertIn(name, replay_fields)

        reward_transform_fields = RewardTransformConfig.__dataclass_fields__
        for name in (
            "mode",
            "missing_total_policy",
            "component_weights",
            "missing_component_policy",
        ):
            self.assertIn(name, reward_transform_fields)

        model_fields = ModelConfig.__dataclass_fields__
        for name in (
            "architecture",
            "input_channels",
            "hidden_size",
            "recurrent_hidden_size",
            "task_embedding_size",
            "optimizer",
            "learning_rate",
            "discount_factor",
            "double_dqn",
            "target_update_frequency",
            "compile",
            "num_actions",
            "task_conditioning",
            "task_feature_size",
        ):
            self.assertIn(name, model_fields)

        train_fields = TrainConfig.__dataclass_fields__
        for name in (
            "algorithm",
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

        ppo_fields = PPOConfig.__dataclass_fields__
        for name in (
            "rollout_steps",
            "minibatch_size",
            "epochs",
            "gae_lambda",
            "clip_range",
            "value_loss_coefficient",
            "entropy_coefficient",
            "normalize_advantages",
            "max_grad_norm",
        ):
            self.assertIn(name, ppo_fields)

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
        self.assertIn("smb_ppo_fast_dev", names)
        self.assertIn("smb_dqn_cpu", names)
        self.assertIn("smb_dqn_mps", names)

        path = config_path("smb_dqn_fast_dev")
        self.assertTrue(path.is_absolute())
        self.assertTrue(path.is_file())

        config = load("smb_dqn_fast_dev")
        self.assertIsInstance(config, MarioRLConfig)
        self.assertEqual("SuperMarioBros-1-1-v0", config.env.id)
        self.assertFalse(config.env.reward_clipping)
        self.assertEqual("env", config.reward_transform.mode)
        self.assertTrue(config.replay.store_reward_info)
        self.assertEqual((84, 84), config.env.image_size)
        self.assertEqual((4, 84, 84), config.replay.state_shape)
        self.assertEqual(AUTO_NUM_ACTIONS, config.model.num_actions)
        self.assertEqual(7, resolve_model_num_actions(config))

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

        actor_critic = load("smb_ppo_fast_dev")
        self.assertEqual("ppo", actor_critic.train.algorithm)
        self.assertEqual("recurrent_actor_critic", actor_critic.model.architecture)
        self.assertTrue(actor_critic.model.task_conditioning)
        self.assertEqual(8, actor_critic.ppo.rollout_steps)

        from_path = load(path)
        self.assertEqual(config, from_path)

    def test_model_num_actions_auto_resolves_from_action_set(self):
        config = parse_cli_config(
            [
                "--config",
                "smb_dqn_fast_dev",
                "--env.action_set",
                "nes",
            ]
        )

        resolved = with_resolved_model_num_actions(config)

        self.assertEqual(AUTO_NUM_ACTIONS, config.model.num_actions)
        self.assertEqual(256, resolved.model.num_actions)

    def test_fixed_model_num_actions_must_match_action_set(self):
        config = parse_cli_config(
            [
                "--config",
                "smb_dqn_fast_dev",
                "--env.action_set",
                "nes",
                "--model.num_actions",
                "7",
            ]
        )

        with self.assertRaisesRegex(ValueError, "model.num_actions=7"):
            with_resolved_model_num_actions(config)


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
                "--reward_transform.mode",
                "component_weights",
                "--reward_transform.component_weights",
                "progress=1,death=-0.5",
                "--reward_transform.missing_component_policy",
                "error",
                "--train.algorithm",
                "ppo",
                "--ppo.rollout_steps",
                "4",
            ]
        )

        self.assertTrue(config.train.fast_dev_run)
        self.assertEqual("SuperMarioBros-1-1-v0", config.env.id)
        self.assertEqual((20, 24), config.env.image_size)
        self.assertTrue(config.task_suite.enabled)
        self.assertEqual({"smb1": 1, "smb3": 2}, config.task_suite.family_weights)
        self.assertEqual("component_weights", config.reward_transform.mode)
        self.assertEqual(
            {"progress": 1.0, "death": -0.5},
            config.reward_transform.component_weights,
        )
        self.assertEqual("error", config.reward_transform.missing_component_policy)
        self.assertEqual("ppo", config.train.algorithm)
        self.assertEqual(4, config.ppo.rollout_steps)

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
