"""PyTorch DQN architecture contract tests."""
from __future__ import annotations

import sys
from dataclasses import replace
from unittest import TestCase

import torch

from mario_rl.config import load, resolve_model_num_actions
from mario_rl.envs import TaskFeatureEncoder
from mario_rl.models import (
    DQN,
    DuelingDQN,
    build_model,
    normalize_observation,
)


class DQNModelTest(TestCase):
    """Validate active DQN modules without legacy framework imports."""

    def test_dqn_accepts_channel_first_tensors(self):
        model = DQN(input_channels=4, num_actions=7, input_shape=(4, 84, 84))
        x = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)

        with torch.no_grad():
            y = model(x)

        self.assertEqual((2, 7), tuple(y.shape))
        self.assertTrue(torch.isfinite(y).all())

    def test_observation_normalization_is_explicit(self):
        x = torch.tensor([0, 128, 255], dtype=torch.uint8)
        y = normalize_observation(x)

        self.assertEqual(torch.float32, y.dtype)
        self.assertTrue(torch.allclose(y, torch.tensor([0.0, 128.0 / 255.0, 1.0])))

    def test_dueling_dqn_combines_value_and_centered_advantage(self):
        value = torch.tensor([[2.0], [0.5]])
        advantage = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])

        q_values = DuelingDQN.combine_value_advantage(value, advantage)

        expected = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        self.assertTrue(torch.allclose(expected, q_values))

    def test_model_factory_reads_typed_config_dimensions(self):
        config = load("smb_dqn_mps")
        model = build_model(config)
        x = torch.zeros(2, *config.replay.state_shape, dtype=torch.uint8)

        with torch.no_grad():
            y = model(x)

        self.assertIsInstance(model, DuelingDQN)
        self.assertEqual((2, resolve_model_num_actions(config)), tuple(y.shape))

    def test_model_factory_auto_sizes_native_nes_action_head(self):
        config = load("smb_dqn_fast_dev")
        config = replace(config, env=replace(config.env, action_set="nes"))
        model = build_model(config)
        x = torch.zeros(2, *config.replay.state_shape, dtype=torch.uint8)

        with torch.no_grad():
            y = model(x)

        self.assertEqual(256, model.num_actions)
        self.assertEqual((2, 256), tuple(y.shape))

    def test_dqn_task_conditioning_preserves_pixel_only_call_path(self):
        encoder = TaskFeatureEncoder()
        model = DQN(
            input_channels=4,
            num_actions=7,
            input_shape=(4, 84, 84),
            task_feature_size=encoder.feature_size,
        )
        x = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
        features = encoder.encode_env_id("SuperMarioBros-1-1-v0").to_tensor()

        with torch.no_grad():
            conditioned = model(x, features)
            default_unknown = model(x)

        self.assertEqual((2, 7), tuple(conditioned.shape))
        self.assertEqual((2, 7), tuple(default_unknown.shape))
        self.assertTrue(torch.isfinite(conditioned).all())

    def test_dueling_dqn_task_conditioned_forward_shape(self):
        encoder = TaskFeatureEncoder()
        model = DuelingDQN(
            input_channels=4,
            num_actions=7,
            input_shape=(4, 84, 84),
            task_feature_size=encoder.feature_size,
        )
        x = torch.zeros(3, 4, 84, 84, dtype=torch.uint8)
        features = encoder.encode_env_id("SuperMarioBros3-1-1-v0").to_tensor()

        with torch.no_grad():
            y = model(x, features)

        self.assertEqual((3, 7), tuple(y.shape))
        self.assertTrue(torch.isfinite(y).all())

    def test_active_model_imports_do_not_load_keras_or_tensorflow(self):
        self.assertNotIn("keras", sys.modules)
        self.assertNotIn("tensorflow", sys.modules)
