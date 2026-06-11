"""Typed config and config CLI contract tests."""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

from mario_rl.config import (
    AUTO_NUM_ACTIONS,
    CUSTOM_PIXEL_PROFILE,
    AuxiliaryLossConfig,
    EnvConfig,
    EvalConfig,
    EvaluationMatrixConfig,
    ExplorationConfig,
    MarioRLConfig,
    ModelConfig,
    PPOConfig,
    ReplayConfig,
    RewardTransformConfig,
    SnapshotCurriculumConfig,
    TaskSuiteConfig,
    TrainConfig,
    TrainerConfig,
    action_space_summary,
    available_configs,
    config_path,
    from_mapping,
    load,
    main,
    parse_cli_config,
    pixel_observation_summary,
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
        self.assertIsInstance(config.exploration, ExplorationConfig)
        self.assertIsInstance(config.auxiliary, AuxiliaryLossConfig)
        self.assertIsInstance(config.evaluation_matrix, EvaluationMatrixConfig)
        self.assertIsInstance(config.ppo, PPOConfig)
        self.assertIsInstance(config.snapshot, SnapshotCurriculumConfig)
        self.assertIn(config.experiment_name, "smb_dqn_fast_dev")
        self.assertEqual("runs", config.save_dir)

        env_fields = EnvConfig.__dataclass_fields__
        for name in (
            "id",
            "render_mode",
            "action_set",
            "macro_actions",
            "macro_action_set",
            "seed",
            "pixel_profile",
            "image_size",
            "frame_stack",
            "reward_clipping",
            "frame_skip",
            "video_enabled",
            "record_statistics",
            "max_episode_steps",
            "no_progress_timeout_steps",
            "stuck_penalty",
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

        auxiliary_fields = AuxiliaryLossConfig.__dataclass_fields__
        for name in ("enabled", "targets", "weights", "head_hidden_size"):
            self.assertIn(name, auxiliary_fields)

        exploration_fields = ExplorationConfig.__dataclass_fields__
        for name in (
            "enabled",
            "method",
            "intrinsic_reward_scale",
            "predictor_learning_rate",
            "normalize_observations",
            "normalize_intrinsic_rewards",
            "intrinsic_reward_clip",
            "warmup_steps",
            "warmup_reward_scale",
            "log_intrinsic_rewards",
            "rnd_embedding_size",
            "rnd_hidden_size",
            "observation_source",
        ):
            self.assertIn(name, exploration_fields)

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
            "num_envs",
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

        snapshot_fields = SnapshotCurriculumConfig.__dataclass_fields__
        for name in (
            "enabled",
            "max_snapshots",
            "capture_interval_steps",
            "sample_probability",
            "min_progress",
            "rank_strategy",
            "tags",
        ):
            self.assertIn(name, snapshot_fields)

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
            "mode",
            "curriculum_frontier_size",
            "curriculum_mastery_window",
            "curriculum_mastery_min_episodes",
            "curriculum_mastery_clear_rate",
            "curriculum_mastery_death_rate",
            "curriculum_mastery_progress",
            "curriculum_lost_levels_prerequisite_family",
            "curriculum_state_path",
        ):
            self.assertIn(name, task_suite_fields)

        matrix_fields = EvaluationMatrixConfig.__dataclass_fields__
        for name in (
            "enabled",
            "game_families",
            "single_stage",
            "splits",
            "include_validated",
            "include_env_ids",
            "exclude_env_ids",
            "max_tasks",
            "include_smb3_catalog",
            "seeds",
            "seed",
            "seed_count",
            "episodes_per_task",
            "summary_name",
            "table_name",
            "video_enabled",
        ):
            self.assertIn(name, matrix_fields)

    def test_packaged_configs_are_discoverable_and_load_typed_objects(self):
        names = available_configs()

        self.assertIn("smb_dqn_fast_dev", names)
        self.assertIn("smb_dqn_prioritized_fast_dev", names)
        self.assertIn("smb_dqn_macbook_gate", names)
        self.assertIn("smb_dqn_task_conditioned_fast_dev", names)
        self.assertIn("smb_dqn_task_suite_fast_dev", names)
        self.assertIn("smb_dqn_eval_matrix_fast_dev", names)
        self.assertIn("smb_ppo_auxiliary_fast_dev", names)
        self.assertIn("smb_ppo_fast_dev", names)
        self.assertIn("smb_ppo_macro_fast_dev", names)
        self.assertIn("smb_ppo_rnd_fast_dev", names)
        self.assertIn("smb_ppo_rgb_fast_dev", names)
        self.assertIn("smb_ppo_rgb_high_fidelity", names)
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
        self.assertEqual("grayscale_84", config.env.pixel_profile)
        self.assertEqual((84, 84), config.env.image_size)
        self.assertEqual((4, 84, 84), config.replay.state_shape)
        self.assertEqual(4, config.model.input_channels)
        self.assertEqual(AUTO_NUM_ACTIONS, config.model.num_actions)
        self.assertEqual(12, resolve_model_num_actions(config))

        prioritized = load("smb_dqn_prioritized_fast_dev")
        self.assertTrue(prioritized.replay.prioritized)
        self.assertEqual(0.6, prioritized.replay.priority_alpha)
        self.assertEqual(0.4, prioritized.replay.priority_beta)
        self.assertEqual((4, 84, 84), prioritized.replay.state_shape)

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

        eval_matrix = load("smb_dqn_eval_matrix_fast_dev")
        self.assertTrue(eval_matrix.evaluation_matrix.enabled)
        self.assertEqual(2, eval_matrix.evaluation_matrix.max_tasks)
        self.assertTrue(eval_matrix.evaluation_matrix.include_smb3_catalog)
        self.assertEqual(
            ("SuperMarioBros-1-1-v0", "SuperMarioBros3-1-1-v0"),
            eval_matrix.evaluation_matrix.include_env_ids,
        )

        actor_critic = load("smb_ppo_fast_dev")
        self.assertEqual("ppo", actor_critic.train.algorithm)
        self.assertEqual("recurrent_actor_critic", actor_critic.model.architecture)
        self.assertTrue(actor_critic.model.task_conditioning)
        self.assertFalse(actor_critic.exploration.enabled)
        self.assertEqual(8, actor_critic.ppo.rollout_steps)
        self.assertTrue(actor_critic.task_suite.enabled)
        self.assertEqual("adaptive", actor_critic.task_suite.mode)
        self.assertEqual(
            "SuperMarioBros-1-1-v0",
            actor_critic.task_suite.include_env_ids[0],
        )

        auxiliary = load("smb_ppo_auxiliary_fast_dev")
        self.assertTrue(auxiliary.auxiliary.enabled)
        self.assertEqual(
            (
                "progress_delta",
                "clear",
                "death",
                "transformed_reward",
                "game_family",
            ),
            auxiliary.auxiliary.targets,
        )
        self.assertEqual(0.5, auxiliary.auxiliary.weights["game_family"])

        rnd = load("smb_ppo_rnd_fast_dev")
        self.assertTrue(rnd.exploration.enabled)
        self.assertEqual("rnd", rnd.exploration.method)
        self.assertEqual("next_observation", rnd.exploration.observation_source)
        self.assertEqual(0.05, rnd.exploration.intrinsic_reward_scale)
        self.assertEqual(1.0, rnd.exploration.intrinsic_reward_clip)

        macro = load("smb_ppo_macro_fast_dev")
        macro_summary = action_space_summary(macro)
        self.assertTrue(macro.env.macro_actions)
        self.assertEqual("conservative", macro.env.macro_action_set)
        self.assertEqual("complex", macro_summary["base_action_set"])
        self.assertEqual(12, macro_summary["base_action_count"])
        self.assertTrue(macro_summary["macro_actions_enabled"])
        self.assertEqual("conservative", macro_summary["macro_action_set"])
        self.assertEqual(macro_summary["action_count"], resolve_model_num_actions(macro))
        self.assertGreater(macro_summary["action_count"], macro_summary["base_action_count"])

        from_path = load(path)
        self.assertEqual(config, from_path)

    def test_packaged_pixel_profiles_resolve_replay_and_model_shapes(self):
        rgb = load("smb_ppo_rgb_fast_dev")

        self.assertFalse(rgb.env.grayscale)
        self.assertEqual("rgb_balanced_90x96", rgb.env.pixel_profile)
        self.assertEqual((90, 96), rgb.env.image_size)
        self.assertEqual((12, 90, 96), rgb.replay.state_shape)
        self.assertEqual(12, rgb.model.input_channels)
        self.assertEqual(2, rgb.ppo.num_envs)

        summary = pixel_observation_summary(rgb)
        self.assertEqual("rgb_balanced_90x96", summary["pixel_profile"])
        self.assertEqual([12, 90, 96], summary["state_shape"])
        self.assertEqual(103680, summary["bytes_per_observation"])

        high = load("smb_ppo_rgb_high_fidelity")
        self.assertEqual("rgb_high_fidelity_120x128", high.env.pixel_profile)
        self.assertEqual((12, 120, 128), high.replay.state_shape)
        self.assertEqual(12, high.model.input_channels)

    def test_pixel_profile_can_fill_derived_shape_defaults(self):
        config = from_mapping(
            {
                "env": {"pixel_profile": "rgb_balanced_90x96"},
                "replay": {"capacity": 16},
                "model": {"architecture": "recurrent_actor_critic"},
            }
        )

        self.assertEqual((90, 96), config.env.image_size)
        self.assertFalse(config.env.grayscale)
        self.assertEqual((12, 90, 96), config.replay.state_shape)
        self.assertEqual(12, config.model.input_channels)

    def test_pixel_shape_mismatches_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "replay.state_shape"):
            from_mapping(
                {
                    "env": {
                        "pixel_profile": "rgb_balanced_90x96",
                        "image_size": [90, 96],
                        "frame_stack": 4,
                        "grayscale": False,
                        "channel_first": True,
                        "interpolation": "area",
                    },
                    "replay": {"state_shape": [4, 90, 96]},
                }
            )

        with self.assertRaisesRegex(ValueError, "model.input_channels"):
            from_mapping(
                {
                    "env": {"pixel_profile": "rgb_balanced_90x96"},
                    "model": {"input_channels": 4},
                }
            )

        with self.assertRaisesRegex(ValueError, "env.image_size"):
            from_mapping(
                {
                    "env": {
                        "pixel_profile": "rgb_balanced_90x96",
                        "image_size": [84, 84],
                    },
                }
            )

    def test_all_packaged_configs_load_with_resolved_pixel_shapes(self):
        for name in available_configs():
            with self.subTest(name=name):
                config = load(name)
                expected_channels = (1 if config.env.grayscale else 3) * int(
                    config.env.frame_stack or 1
                )
                self.assertEqual(expected_channels, config.model.input_channels)
                self.assertEqual(expected_channels, config.replay.state_shape[0])
                self.assertEqual(config.env.image_size, config.replay.state_shape[1:])

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

    def test_model_num_actions_auto_resolves_from_macro_actions(self):
        config = parse_cli_config(
            [
                "--config",
                "smb_ppo_fast_dev",
                "--env.macro_actions",
                "true",
            ]
        )

        resolved = with_resolved_model_num_actions(config)
        summary = action_space_summary(config)

        self.assertEqual(AUTO_NUM_ACTIONS, config.model.num_actions)
        self.assertTrue(summary["macro_actions_enabled"])
        self.assertEqual(summary["action_count"], resolved.model.num_actions)
        self.assertGreater(resolved.model.num_actions, summary["base_action_count"])

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
                "--env.grayscale",
                "false",
                "--task_suite.enabled",
                "true",
                "--task_suite.family_weights",
                "smb1=1,smb3=2",
                "--evaluation_matrix.enabled",
                "true",
                "--evaluation_matrix.max_tasks",
                "2",
                "--evaluation_matrix.seeds",
                "7,8",
                "--reward_transform.mode",
                "component_weights",
                "--reward_transform.component_weights",
                "progress=1,death=-0.5",
                "--reward_transform.missing_component_policy",
                "error",
                "--auxiliary.enabled",
                "true",
                "--auxiliary.targets",
                "clear,death,game_family",
                "--auxiliary.weights",
                "clear=0.25,game_family=2",
                "--exploration.enabled",
                "true",
                "--exploration.intrinsic_reward_scale",
                "0.1",
                "--exploration.intrinsic_reward_clip",
                "0.75",
                "--train.algorithm",
                "ppo",
                "--ppo.rollout_steps",
                "4",
            ]
        )

        self.assertTrue(config.train.fast_dev_run)
        self.assertEqual("SuperMarioBros-1-1-v0", config.env.id)
        self.assertEqual(CUSTOM_PIXEL_PROFILE, config.env.pixel_profile)
        self.assertEqual((20, 24), config.env.image_size)
        self.assertFalse(config.env.grayscale)
        self.assertEqual((12, 20, 24), config.replay.state_shape)
        self.assertEqual(12, config.model.input_channels)
        self.assertTrue(config.task_suite.enabled)
        self.assertEqual({"smb1": 1, "smb3": 2}, config.task_suite.family_weights)
        self.assertTrue(config.evaluation_matrix.enabled)
        self.assertEqual(2, config.evaluation_matrix.max_tasks)
        self.assertEqual((7, 8), config.evaluation_matrix.seeds)
        self.assertEqual("component_weights", config.reward_transform.mode)
        self.assertEqual(
            {"progress": 1.0, "death": -0.5},
            config.reward_transform.component_weights,
        )
        self.assertEqual("error", config.reward_transform.missing_component_policy)
        self.assertTrue(config.auxiliary.enabled)
        self.assertEqual(("clear", "death", "game_family"), config.auxiliary.targets)
        self.assertEqual(
            {"clear": 0.25, "game_family": 2.0},
            config.auxiliary.weights,
        )
        self.assertTrue(config.exploration.enabled)
        self.assertEqual(0.1, config.exploration.intrinsic_reward_scale)
        self.assertEqual(0.75, config.exploration.intrinsic_reward_clip)
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
