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
from mario_rl.config import SnapshotCurriculumConfig
from mario_rl.envs import TaskSuiteConfig
from mario_rl.exploration import ExplorationConfig
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
            self.assertEqual("complex", payload["action_set"])
            self.assertEqual(12, payload["action_count"])
            self.assertFalse(payload["native_action_space"])
            self.assertGreaterEqual(payload["global_step"], 1)
            self.assertEqual(config.train.max_steps, payload["env_frames"])
            self.assertTrue(Path(payload["checkpoint"]).is_file())
            self.assertTrue(Path(payload["metrics"]).is_file())
            self.assertTrue(Path(payload["metrics_json"]).is_file())
            with Path(payload["metrics"]).open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))[-1]
            self.assertEqual("complex", metrics["action_set"])
            self.assertEqual("12", metrics["action_count"])
            self.assertIn("metric_episode_count", metrics)
            self.assertIn("clear_rate", metrics)
            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())
            self.assertEqual("train", structured_metrics["command"])
            self.assertEqual([4, 84, 84], structured_metrics["pixel_observation"]["state_shape"])
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

    def test_train_run_writes_prioritized_replay_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            config = replace(
                config,
                replay=replace(config.replay, prioritized=True, warmup=1),
                train=replace(config.train, max_steps=4),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            with Path(payload["metrics"]).open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))[-1]
            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())
            replay = structured_metrics["replay"]

            self.assertTrue(replay["prioritized"])
            self.assertGreaterEqual(replay["priority_updates"], 1)
            self.assertGreater(replay["max_priority"], 0.0)
            self.assertEqual("True", metrics["replay_prioritized"])
            self.assertGreaterEqual(int(metrics["replay_priority_updates"]), 1)

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
            self.assertEqual(
                config.train.max_steps * config.ppo.rollout_steps * config.ppo.num_envs,
                payload["env_frames"],
            )
            self.assertTrue(Path(payload["checkpoint"]).is_file())
            self.assertTrue(Path(payload["metrics_json"]).is_file())
            with Path(payload["metrics"]).open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))[-1]
            self.assertIn("ppo_policy_loss", metrics)
            self.assertEqual(str(config.ppo.num_envs), metrics["ppo_num_envs"])
            self.assertIn("auxiliary_loss", metrics)
            self.assertIn("game_family", metrics["auxiliary_losses_json"])
            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())
            self.assertEqual("ppo", structured_metrics["algorithm"])
            self.assertEqual([4, 84, 84], structured_metrics["pixel_observation"]["state_shape"])
            self.assertIn("ppo", structured_metrics)
            self.assertEqual(config.ppo.num_envs, structured_metrics["ppo"]["num_envs"])
            self.assertIn("auxiliary", structured_metrics)
            self.assertTrue(structured_metrics["auxiliary"]["enabled"])
            self.assertIn("game_family", structured_metrics["auxiliary"]["losses"])
            self.assertIn("global", structured_metrics)
            self.assertIn("fake_ppo_lightning", Path(payload["resolved_config"]).read_text())

    def test_train_run_with_macro_actions_writes_macro_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                experiment_name="fake_ppo_macro_lightning",
                env=replace(
                    config.env,
                    macro_actions=True,
                    macro_action_set="conservative",
                ),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual("train", payload["command"])
            self.assertEqual("ppo", payload["algorithm"])
            self.assertEqual("complex", payload["base_action_set"])
            self.assertEqual(12, payload["base_action_count"])
            self.assertTrue(payload["macro_actions_enabled"])
            self.assertEqual("conservative", payload["macro_action_set"])
            self.assertEqual(payload["action_count"], payload["macro_action_count"])
            self.assertGreater(payload["action_count"], payload["base_action_count"])
            self.assertTrue(
                any(action["name"] == "run_jump" for action in payload["macro_actions"])
            )
            self.assertEqual(config.env.frame_skip, payload["macro_frame_skip"])
            self.assertGreaterEqual(payload["global_step"], 1)

            with Path(payload["metrics"]).open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))[-1]
            self.assertEqual("True", metrics["macro_actions_enabled"])
            self.assertEqual("conservative", metrics["macro_action_set"])
            self.assertIn("run_jump", metrics["macro_actions_json"])

            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())
            self.assertTrue(structured_metrics["macro_actions_enabled"])
            self.assertEqual(payload["action_count"], structured_metrics["action_count"])
            resolved_config_text = Path(payload["resolved_config"]).read_text()
            self.assertIn("macro_actions: true", resolved_config_text)
            self.assertIn("macro_action_set: conservative", resolved_config_text)

    def test_train_run_writes_rnd_exploration_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                exploration=ExplorationConfig(
                    enabled=True,
                    intrinsic_reward_scale=0.05,
                    intrinsic_reward_clip=0.5,
                    rnd_embedding_size=16,
                    rnd_hidden_size=32,
                ),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertTrue(payload["exploration"]["exploration_enabled"])
            self.assertEqual(
                "next_observation",
                payload["exploration"]["exploration_observation_source"],
            )
            with Path(payload["metrics"]).open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))[-1]
            self.assertEqual("True", metrics["exploration_enabled"])
            self.assertEqual("rnd", metrics["exploration_method"])
            self.assertEqual("next_observation", metrics["exploration_observation_source"])
            self.assertGreater(float(metrics["intrinsic_reward_mean"]), 0.0)
            self.assertGreater(float(metrics["rnd_loss"]), 0.0)
            self.assertGreater(float(metrics["rnd_predictor_grad_norm"]), 0.0)
            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())
            exploration = structured_metrics["exploration"]
            self.assertTrue(exploration["exploration_enabled"])
            self.assertEqual(
                "next_observation",
                exploration["exploration_observation_source"],
            )
            self.assertGreater(exploration["intrinsic_reward_mean"], 0.0)
            self.assertGreater(exploration["rnd_loss"], 0.0)

    def test_train_run_writes_adaptive_curriculum_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            config = replace(
                config,
                task_suite=TaskSuiteConfig(
                    enabled=True,
                    mode="adaptive",
                    include_env_ids=(
                        "SuperMarioBros-1-1-v0",
                        "SuperMarioBros-1-2-v0",
                        "SuperMarioBros2-1-1-v0",
                    ),
                    single_stage=True,
                    seed=7,
                    curriculum_mastery_min_episodes=1,
                    curriculum_mastery_clear_rate=1.0,
                    curriculum_mastery_death_rate=0.0,
                ),
                replay=replace(config.replay, warmup=1),
                train=replace(config.train, max_steps=5),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            self.assertTrue(Path(payload["curriculum_state"]).is_file())
            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())
            curriculum = structured_metrics["curriculum"]
            state = curriculum["state"]

            self.assertEqual("adaptive", curriculum["metadata"]["mode"])
            self.assertEqual(
                "SuperMarioBros-1-1-v0",
                state["episode_task_env_ids"]["0"],
            )
            self.assertTrue(
                any(
                    record["env_id"] == "SuperMarioBros-1-1-v0"
                    and record["mastered"]
                    for record in state["records"]
                )
            )
            self.assertGreaterEqual(curriculum["counts"]["active"], 1)

    def test_train_run_writes_snapshot_metadata_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            config = replace(
                config,
                snapshot=SnapshotCurriculumConfig(
                    enabled=True,
                    max_snapshots=8,
                    capture_interval_steps=1,
                    sample_probability=1.0,
                    tags=("fake-train",),
                ),
                replay=replace(config.replay, warmup=1),
                train=replace(config.train, max_steps=5),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, run(config, env_factory=fake_env_factory))

            payload = json.loads(output.getvalue().splitlines()[-1])
            snapshot_path = Path(payload["snapshot_metadata"])
            self.assertTrue(snapshot_path.is_file())
            snapshot_text = snapshot_path.read_text(encoding="utf-8")
            snapshot_payload = json.loads(snapshot_text)
            structured_metrics = json.loads(Path(payload["metrics_json"]).read_text())

            self.assertGreaterEqual(snapshot_payload["counts"]["captured"], 1)
            self.assertGreaterEqual(snapshot_payload["counts"]["restored"], 1)
            self.assertGreaterEqual(
                structured_metrics["global"]["snapshot_start_count"],
                1,
            )
            self.assertIn("snapshots", structured_metrics)
            self.assertFalse(
                snapshot_payload["serialization"]["artifact_contains_rom_bytes"]
            )
            self.assertNotIn("native_snapshot", snapshot_text)
            self.assertNotIn('"observation"', snapshot_text)
