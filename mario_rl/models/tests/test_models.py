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
    RecurrentActorCritic,
    build_model,
    normalize_observation,
    reset_recurrent_state,
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

    def test_model_factory_builds_rgb_recurrent_actor_critic(self):
        config = load("smb_ppo_rgb_fast_dev")
        model = build_model(config)
        x = torch.zeros(1, *config.replay.state_shape, dtype=torch.uint8)
        hidden = model.initial_state(batch_size=1)

        with torch.no_grad():
            output = model(x, hidden)

        self.assertIsInstance(model, RecurrentActorCritic)
        self.assertEqual((12, 90, 96), model.input_shape)
        self.assertEqual(12, model.features[0].in_channels)
        self.assertEqual((1, resolve_model_num_actions(config)), tuple(output.policy_logits.shape))
        self.assertEqual((1,), tuple(output.value.shape))

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

    def test_recurrent_actor_critic_forward_shape_without_task_conditioning(self):
        model = RecurrentActorCritic(
            input_channels=4,
            num_actions=7,
            input_shape=(4, 84, 84),
            hidden_size=64,
            recurrent_hidden_size=32,
        )
        x = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
        hidden = model.initial_state(batch_size=2)

        with torch.no_grad():
            output = model(x, hidden)

        self.assertEqual((2, 7), tuple(output.policy_logits.shape))
        self.assertEqual((2,), tuple(output.value.shape))
        self.assertEqual((1, 2, 32), tuple(output.hidden_state.shape))
        self.assertTrue(torch.isfinite(output.policy_logits).all())
        self.assertTrue(torch.isfinite(output.value).all())

    def test_recurrent_actor_critic_forward_shape_with_task_conditioning(self):
        encoder = TaskFeatureEncoder()
        model = RecurrentActorCritic(
            input_channels=4,
            num_actions=7,
            input_shape=(4, 84, 84),
            hidden_size=64,
            recurrent_hidden_size=32,
            task_feature_size=encoder.feature_size,
            task_embedding_size=8,
        )
        x = torch.zeros(3, 2, 4, 84, 84, dtype=torch.uint8)
        features = torch.stack(
            [
                encoder.encode_env_id("SuperMarioBros-1-1-v0").to_tensor(),
                encoder.encode_env_id("SuperMarioBros3-1-1-v0").to_tensor(),
            ]
        )

        with torch.no_grad():
            output = model(x, task_features=features)

        self.assertEqual((3, 2, 7), tuple(output.policy_logits.shape))
        self.assertEqual((3, 2), tuple(output.value.shape))
        self.assertEqual((1, 2, 32), tuple(output.hidden_state.shape))

    def test_recurrent_actor_critic_auxiliary_head_shapes(self):
        model = RecurrentActorCritic(
            input_channels=4,
            num_actions=7,
            input_shape=(4, 84, 84),
            hidden_size=64,
            recurrent_hidden_size=32,
            auxiliary_outputs={"clear": 1, "game_family": 5},
            auxiliary_hidden_size=16,
        )
        x = torch.zeros(4, 2, 4, 84, 84, dtype=torch.uint8)

        with torch.no_grad():
            output = model(x)

        self.assertEqual((4, 2), tuple(output.auxiliary["clear"].shape))
        self.assertEqual((4, 2, 5), tuple(output.auxiliary["game_family"].shape))
        self.assertTrue(torch.isfinite(output.auxiliary["clear"]).all())
        self.assertTrue(torch.isfinite(output.auxiliary["game_family"]).all())

    def test_recurrent_hidden_state_reset_masks_completed_episodes(self):
        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

        reset = reset_recurrent_state(hidden, torch.tensor([False, True, False]))

        self.assertTrue(torch.equal(hidden[:, 0], reset[:, 0]))
        self.assertTrue(torch.equal(torch.zeros(1, 4), reset[:, 1]))
        self.assertTrue(torch.equal(hidden[:, 2], reset[:, 2]))

    def test_active_model_imports_do_not_load_keras_or_tensorflow(self):
        self.assertNotIn("keras", sys.modules)
        self.assertNotIn("tensorflow", sys.modules)
