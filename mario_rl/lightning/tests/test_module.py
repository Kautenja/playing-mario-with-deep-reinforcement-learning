"""Lightning DQN training module contract tests."""
from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer

from mario_rl.auxiliary import AuxiliaryLossConfig
from mario_rl.config import SnapshotCurriculumConfig, load
from mario_rl.envs import UNKNOWN_TASK_VALUE
from mario_rl.envs import TaskSuite, TaskSuiteConfig
from mario_rl.exploration import ExplorationConfig
from mario_rl.lightning import DQNLightningModule, PPOLightningModule, trainer_accelerator
from mario_rl.rewards import RewardTransformConfig
from mario_rl.tests.fakes import (
    FakeMarioEnv,
    fake_env_factory,
    tiny_ppo_config,
    tiny_training_config,
)


class LightningModuleTest(TestCase):
    """Validate the active Lightning DQN integration path."""

    def test_module_owns_networks_optimizer_replay_and_schedule(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            module = DQNLightningModule(config, env_factory=fake_env_factory)

            self.assertIsInstance(module, LightningModule)
            self.assertFalse(module.automatic_optimization)
            self.assertEqual(config.replay.capacity, module.replay.capacity)
            self.assertEqual(config.epsilon.start, module.epsilon_schedule.start)

            optimizer = module.configure_optimizers()
            self.assertEqual(config.model.learning_rate, optimizer.param_groups[0]["lr"])
            for online, target in zip(module.q_network.parameters(), module.target_q_network.parameters()):
                self.assertTrue(torch.equal(online, target))

    def test_fake_env_fast_dev_run_trains_and_logs_finite_state(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            module = DQNLightningModule(config, env_factory=fake_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)

            self.assertEqual(config.train.max_steps, module.env_frames)
            self.assertGreaterEqual(module.training_updates, 1)
            self.assertTrue(math.isfinite(module.last_loss))
            self.assertGreaterEqual(len(module.replay), config.replay.warmup)
            self.assertIn("train/clear_rate", trainer.callback_metrics)
            self.assertIn("train/death_rate", trainer.callback_metrics)
            self.assertIn("train/max_progress", trainer.callback_metrics)
            self.assertGreaterEqual(
                module.metrics.global_summary(include_active=True).step_count,
                1,
            )

    def test_component_reward_transform_trains_with_fake_reward_components(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            config = replace(
                config,
                reward_transform=RewardTransformConfig(
                    mode="component_weights",
                    component_weights={"progress": 2.0},
                ),
                replay=replace(config.replay, warmup=1),
                train=replace(config.train, max_steps=4),
            )
            module = DQNLightningModule(config, env_factory=fake_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)
            batch = module.replay.sample(len(module.replay))

            self.assertIsNotNone(batch.unclipped_reward)
            self.assertTrue(np.allclose(batch.reward, batch.unclipped_reward * 2.0))
            self.assertGreaterEqual(module.training_updates, 1)

    def test_checkpoint_round_trip_restores_weights_and_schedule_state(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            module = DQNLightningModule(config, env_factory=fake_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )
            trainer.fit(module)
            checkpoint = Path(tmpdir) / "round-trip.ckpt"
            trainer.save_checkpoint(checkpoint)

            state = torch.zeros(1, *config.replay.state_shape, dtype=torch.uint8)
            with torch.no_grad():
                expected = module.q_network(state)

            loaded = DQNLightningModule.load_from_checkpoint(
                checkpoint,
                config=config,
                env_factory=fake_env_factory,
                map_location="cpu",
            )
            with torch.no_grad():
                actual = loaded.q_network(state)

            self.assertTrue(torch.allclose(expected, actual))
            self.assertEqual(module.env_frames, loaded.env_frames)
            self.assertEqual(module.epsilon_schedule.current_step, loaded.epsilon_schedule.current_step)

    def test_checkpoint_round_trip_restores_adaptive_curriculum_state(self):
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
                    ),
                    single_stage=True,
                    seed=13,
                    curriculum_mastery_min_episodes=1,
                    curriculum_mastery_clear_rate=1.0,
                    curriculum_mastery_death_rate=0.0,
                ),
                replay=replace(config.replay, warmup=1),
                train=replace(config.train, max_steps=5),
            )
            module = DQNLightningModule(config, env_factory=fake_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )
            trainer.fit(module)
            checkpoint = Path(tmpdir) / "adaptive-round-trip.ckpt"
            trainer.save_checkpoint(checkpoint)

            loaded = DQNLightningModule.load_from_checkpoint(
                checkpoint,
                config=config,
                env_factory=fake_env_factory,
                map_location="cpu",
            )

            records = {
                record["env_id"]: record
                for record in loaded.task_suite.state_dict()["records"]
            }
            self.assertTrue(records["SuperMarioBros-1-1-v0"]["mastered"])
            self.assertEqual(
                "SuperMarioBros-1-2-v0",
                loaded.task_suite.task_for_episode(loaded.episodes).env_id,
            )

    def test_task_conditioning_trains_with_fake_unknown_task_metadata(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            config = replace(
                config,
                model=replace(config.model, task_conditioning=True, task_feature_size=0),
            )
            module = DQNLightningModule(config, env_factory=fake_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)
            batch = module.replay.sample(1)

            encoded = module.task_encoder.encode_env_id(config.env.id)
            self.assertEqual(UNKNOWN_TASK_VALUE, encoded.task_id)
            self.assertIsNotNone(batch.task_features)
            self.assertEqual(module.q_network.task_feature_size, batch.task_features.shape[1])
            self.assertGreaterEqual(module.training_updates, 1)

    def test_task_suite_switches_fake_envs_at_episode_boundaries(self):
        with TemporaryDirectory() as tmpdir:
            seen_env_ids = []

            def tracking_env_factory(config):
                seen_env_ids.append(config.env.id)
                return FakeMarioEnv(episode_length=1)

            config = tiny_training_config(tmpdir)
            suite_config = TaskSuiteConfig(
                enabled=True,
                include_env_ids=("SuperMarioBros-1-1-v0", "SuperMarioBros3-1-1-v0"),
                single_stage=True,
                seed=2,
                switch_interval_episodes=1,
            )
            config = replace(
                config,
                env=replace(config.env, id="SuperMarioBros-1-1-v0"),
                task_suite=suite_config,
                replay=replace(config.replay, warmup=1),
                train=replace(config.train, max_steps=6),
            )
            module = DQNLightningModule(config, env_factory=tracking_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)

            expected = []
            last_env_id = None
            suite = TaskSuite(suite_config)
            for episode in range(module.episodes + 1):
                env_id = suite.task_for_episode(episode).env_id
                if env_id != last_env_id:
                    expected.append(env_id)
                    last_env_id = env_id
            self.assertEqual(expected, seen_env_ids)
            self.assertGreater(len(set(seen_env_ids)), 1)

    def test_single_env_path_keeps_configured_env_id(self):
        with TemporaryDirectory() as tmpdir:
            seen_env_ids = []

            def tracking_env_factory(config):
                seen_env_ids.append(config.env.id)
                return FakeMarioEnv(episode_length=1)

            config = tiny_training_config(tmpdir)
            config = replace(
                config,
                replay=replace(config.replay, warmup=1),
                train=replace(config.train, max_steps=4),
            )
            module = DQNLightningModule(config, env_factory=tracking_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)

            self.assertFalse(config.task_suite.enabled)
            self.assertEqual([config.env.id], seen_env_ids)


class PPOLightningModuleTest(TestCase):
    """Validate recurrent actor-critic PPO integration."""

    def test_fake_env_ppo_run_optimizes_and_logs_finite_state(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            module = PPOLightningModule(config, env_factory=fake_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)

            self.assertEqual(config.train.max_steps * config.ppo.rollout_steps, module.env_frames)
            self.assertGreaterEqual(module.training_updates, 1)
            self.assertTrue(math.isfinite(module.last_loss))
            self.assertTrue(math.isfinite(module.last_policy_loss))
            self.assertTrue(math.isfinite(module.last_value_loss))
            self.assertGreaterEqual(module.episodes, 1)
            self.assertIn("train/ppo_policy_loss", trainer.callback_metrics)
            self.assertIn("train/ppo_num_envs", trainer.callback_metrics)
            self.assertIn("train/clear_rate", trainer.callback_metrics)

    def test_ppo_recurrent_state_is_zeroed_after_terminal_step(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(config, ppo=replace(config.ppo, rollout_steps=4))
            module = PPOLightningModule(config, env_factory=fake_env_factory)
            module._ensure_env()
            rollout = module._collect_rollout()

            self.assertTrue(rollout.terminated[-1, 0])
            self.assertIsNotNone(module._hidden_state)
            self.assertTrue(torch.equal(module._hidden_state, torch.zeros_like(module._hidden_state)))

    def test_vectorized_ppo_rollout_batches_slots_and_resets_only_finished_hidden(self):
        with TemporaryDirectory() as tmpdir:
            created = 0

            def env_factory(config):
                nonlocal created
                episode_length = 1 if created == 0 else 99
                created += 1
                return FakeMarioEnv(episode_length=episode_length, env_id=config.env.id)

            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                ppo=replace(config.ppo, num_envs=2, rollout_steps=3),
            )
            module = PPOLightningModule(config, env_factory=env_factory)
            module._ensure_env()
            rollout = module._collect_rollout()

            self.assertEqual(2, rollout.num_envs)
            self.assertEqual((3, 2), rollout.actions.shape)
            self.assertTrue(np.all(rollout.terminated[:, 0]))
            self.assertFalse(np.any(rollout.terminated[:, 1]))
            self.assertIsNotNone(module._hidden_state)
            self.assertTrue(
                torch.equal(
                    module._hidden_state[:, 0],
                    torch.zeros_like(module._hidden_state[:, 0]),
                )
            )
            self.assertFalse(
                torch.equal(
                    module._hidden_state[:, 1],
                    torch.zeros_like(module._hidden_state[:, 1]),
                )
            )
            self.assertEqual(3, module.episodes)
            self.assertEqual(6, module.metrics.global_summary(include_active=True).step_count)

    def test_vectorized_ppo_rgb_rollout_storage_uses_resolved_state_shape(self):
        with TemporaryDirectory() as tmpdir:
            config = load("smb_ppo_rgb_fast_dev")
            config = replace(
                config,
                save_dir=tmpdir,
                env=replace(config.env, id="FakeMario-v0"),
                task_suite=TaskSuiteConfig(enabled=False),
            )
            module = PPOLightningModule(config, env_factory=fake_env_factory)
            rollout = module._new_rollout_storage()

            self.assertEqual((12, 90, 96), rollout.observation_shape)
            self.assertEqual((8, 2, 12, 90, 96), rollout.observations.shape)
            self.assertEqual((12, 90, 96), module.policy.input_shape)
            self.assertEqual(12, module.policy.features[0].in_channels)

    def test_vectorized_ppo_task_suite_assigns_initial_slots_independently(self):
        class _Task:
            def __init__(self, env_id):
                self.env_id = env_id

        class _CyclingSuite:
            def task_for_episode(self, episode):
                return _Task(("FakeMario-A-v0", "FakeMario-B-v0")[int(episode) % 2])

        with TemporaryDirectory() as tmpdir:
            seen_env_ids = []

            def env_factory(config):
                seen_env_ids.append(config.env.id)
                return FakeMarioEnv(episode_length=4, env_id=config.env.id)

            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                ppo=replace(config.ppo, num_envs=2, rollout_steps=1),
            )
            module = PPOLightningModule(config, env_factory=env_factory)
            module.task_suite = _CyclingSuite()

            module._ensure_env()

            self.assertEqual(["FakeMario-A-v0", "FakeMario-B-v0"], seen_env_ids)

    def test_vectorized_ppo_snapshot_curriculum_restores_slots(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                snapshot=SnapshotCurriculumConfig(
                    enabled=True,
                    max_snapshots=8,
                    capture_interval_steps=1,
                    sample_probability=1.0,
                ),
                ppo=replace(config.ppo, num_envs=2, rollout_steps=4),
                train=replace(config.train, max_steps=2),
            )
            module = PPOLightningModule(config, env_factory=fake_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)

            payload = module.snapshot_payload()
            metrics = module.metrics_payload(include_active=True)

            self.assertIsNotNone(payload)
            self.assertGreaterEqual(payload["counts"]["captured"], 1)
            self.assertGreaterEqual(payload["counts"]["restored"], 1)
            self.assertGreaterEqual(metrics["global"]["snapshot_start_count"], 1)
            captured_slots = {
                entry["seed_lineage"][-1]
                for entry in payload["entries"]
                if entry["seed_lineage"]
            }
            self.assertEqual({"0", "1"}, captured_slots)

    def test_fake_env_ppo_run_trains_with_auxiliary_losses(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                auxiliary=AuxiliaryLossConfig(
                    enabled=True,
                    targets=(
                        "progress_delta",
                        "clear",
                        "death",
                        "transformed_reward",
                        "game_family",
                    ),
                    weights={
                        "progress_delta": 0.1,
                        "clear": 0.25,
                        "death": 0.25,
                        "transformed_reward": 0.1,
                        "game_family": 0.5,
                    },
                    head_hidden_size=16,
                ),
            )
            module = PPOLightningModule(config, env_factory=fake_env_factory)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)

            self.assertGreaterEqual(module.training_updates, 1)
            self.assertTrue(math.isfinite(module.last_auxiliary_loss))
            self.assertGreater(module.last_auxiliary_valid_counts["progress_delta"], 0)
            self.assertGreater(module.last_auxiliary_valid_counts["game_family"], 0)
            self.assertIn("train/auxiliary_loss", trainer.callback_metrics)
            self.assertIn("train/auxiliary_game_family_loss", trainer.callback_metrics)

    def test_fake_env_ppo_run_trains_rnd_predictor_only(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            config = replace(
                config,
                exploration=ExplorationConfig(
                    enabled=True,
                    intrinsic_reward_scale=0.05,
                    rnd_embedding_size=16,
                    rnd_hidden_size=32,
                ),
            )
            module = PPOLightningModule(config, env_factory=fake_env_factory)
            self.assertIsNotNone(module.rnd)
            assert module.rnd is not None
            target_before = {
                name: parameter.detach().clone()
                for name, parameter in module.rnd.target.named_parameters()
            }
            predictor_before = {
                name: parameter.detach().clone()
                for name, parameter in module.rnd.predictor.named_parameters()
            }
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                max_steps=-1,
                limit_train_batches=config.train.max_steps,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            trainer.fit(module)

            self.assertGreater(module.last_intrinsic_reward_mean, 0.0)
            self.assertGreater(module.last_rnd_loss, 0.0)
            self.assertGreater(module.last_rnd_predictor_grad_norm, 0.0)
            self.assertIn("train/intrinsic_reward_mean", trainer.callback_metrics)
            self.assertIn("train/rnd_loss", trainer.callback_metrics)
            for name, parameter in module.rnd.target.named_parameters():
                self.assertTrue(torch.equal(target_before[name], parameter.detach()))
                self.assertFalse(parameter.requires_grad)
            self.assertTrue(
                any(
                    not torch.equal(predictor_before[name], parameter.detach())
                    for name, parameter in module.rnd.predictor.named_parameters()
                )
            )


class LightningDeviceSelectionTest(TestCase):
    """Validate config-level accelerator mapping and device placement."""

    def test_cuda_alias_maps_to_lightning_gpu_accelerator(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            config = replace(config, train=replace(config.train, accelerator="cuda"))
            self.assertEqual("gpu", trainer_accelerator(config))

    def test_cpu_model_and_tensors_stay_on_cpu(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            module = DQNLightningModule(config, env_factory=fake_env_factory).to("cpu")
            state = torch.zeros(1, *config.replay.state_shape, dtype=torch.uint8, device="cpu")
            with torch.no_grad():
                output = module.q_network(state)
            self.assertEqual(torch.device("cpu"), output.device)

    @skipIf(not torch.backends.mps.is_available(), "MPS is not available")
    def test_mps_model_and_tensors_move_to_mps_when_available(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            module = DQNLightningModule(config, env_factory=fake_env_factory).to("mps")
            state = torch.zeros(1, *config.replay.state_shape, dtype=torch.uint8, device="mps")
            with torch.no_grad():
                output = module.q_network(state)
            self.assertEqual("mps", output.device.type)

    @skipIf(
        not (torch.cuda.is_available() and os.environ.get("MARIO_RL_RUN_CUDA_SMOKE") == "1"),
        "CUDA smoke is not enabled",
    )
    def test_cuda_model_and_tensors_move_to_cuda_when_enabled(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_training_config(tmpdir)
            module = DQNLightningModule(config, env_factory=fake_env_factory).to("cuda")
            state = torch.zeros(1, *config.replay.state_shape, dtype=torch.uint8, device="cuda")
            with torch.no_grad():
                output = module.q_network(state)
            self.assertEqual(torch.device("cuda:0"), output.device)
