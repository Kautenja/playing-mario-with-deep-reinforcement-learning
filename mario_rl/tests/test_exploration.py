"""Pixel-only curiosity exploration contract tests."""
from __future__ import annotations

import inspect
from unittest import TestCase

import torch

from mario_rl.exploration import (
    RND_OBSERVATION_SOURCE,
    ExplorationConfig,
    RandomNetworkDistillation,
    exploration_summary,
)


class RandomNetworkDistillationTest(TestCase):
    """Validate RND rewards without privileged game state."""

    def test_intrinsic_rewards_are_finite_clipped_and_deterministic(self):
        observation = torch.arange(2 * 4 * 84 * 84, dtype=torch.uint8).reshape(
            2,
            4,
            84,
            84,
        )
        torch.manual_seed(123)
        first = RandomNetworkDistillation(
            input_shape=(4, 84, 84),
            embedding_size=16,
            hidden_size=32,
            intrinsic_reward_clip=0.5,
        )
        first_reward = first(observation)

        torch.manual_seed(123)
        second = RandomNetworkDistillation(
            input_shape=(4, 84, 84),
            embedding_size=16,
            hidden_size=32,
            intrinsic_reward_clip=0.5,
        )
        second_reward = second(observation)

        self.assertTrue(torch.isfinite(first_reward.intrinsic_reward).all())
        self.assertTrue(torch.all(first_reward.intrinsic_reward >= 0.0))
        self.assertTrue(torch.all(first_reward.intrinsic_reward <= 0.5))
        self.assertTrue(
            torch.allclose(
                first_reward.intrinsic_reward,
                second_reward.intrinsic_reward,
            )
        )
        self.assertTrue(torch.allclose(first_reward.raw_error, second_reward.raw_error))

    def test_predictor_receives_gradients_and_target_is_frozen(self):
        torch.manual_seed(7)
        rnd = RandomNetworkDistillation(
            input_shape=(4, 84, 84),
            embedding_size=16,
            hidden_size=32,
        )
        observation = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)

        reward = rnd(observation)
        reward.predictor_loss.backward()

        self.assertTrue(all(not parameter.requires_grad for parameter in rnd.target.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in rnd.target.parameters()))
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.any(parameter.grad.detach() != 0.0))
                for parameter in rnd.predictor.parameters()
            )
        )

    def test_rnd_forward_accepts_only_pixel_observations(self):
        signature = inspect.signature(RandomNetworkDistillation.forward)

        self.assertEqual(("self", "observation"), tuple(signature.parameters))

    def test_exploration_summary_documents_next_observation_source(self):
        config = ExplorationConfig(enabled=True)
        summary = exploration_summary(config)

        self.assertEqual(RND_OBSERVATION_SOURCE, config.observation_source)
        self.assertEqual(
            RND_OBSERVATION_SOURCE,
            summary["exploration_observation_source"],
        )
        self.assertTrue(summary["exploration_enabled"])
