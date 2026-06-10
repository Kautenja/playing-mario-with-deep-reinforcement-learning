"""Lightning DQN training module contract tests."""
from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

import torch
from lightning.pytorch import LightningModule, Trainer

from mario_rl.envs import UNKNOWN_TASK_VALUE
from mario_rl.envs import TaskSuite, TaskSuiteConfig
from mario_rl.lightning import DQNLightningModule, trainer_accelerator
from mario_rl.tests.fakes import FakeMarioEnv, fake_env_factory, tiny_training_config


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
