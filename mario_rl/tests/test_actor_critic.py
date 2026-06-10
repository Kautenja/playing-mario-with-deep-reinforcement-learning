"""Actor-critic rollout storage contract tests."""
from __future__ import annotations

from unittest import TestCase

import numpy as np
import torch

from mario_rl.actor_critic import RolloutStorage


class RolloutStorageTest(TestCase):
    """Validate recurrent rollout insertion, GAE, and minibatches."""

    def test_rollout_storage_insertion_minibatches_and_gae(self):
        storage = RolloutStorage(
            rollout_steps=3,
            num_envs=1,
            observation_shape=(4, 84, 84),
            hidden_state_shape=(1, 8),
            task_feature_shape=(5,),
            seed=123,
        )
        observation = np.zeros((4, 84, 84), dtype=np.uint8)
        hidden = np.ones((1, 1, 8), dtype=np.float32)
        features = np.arange(5, dtype=np.float32)
        for step in range(3):
            storage.insert(
                observation + step,
                step % 2,
                -0.5,
                1.0,
                terminated=step == 2,
                truncated=False,
                value=0.5,
                hidden_state=hidden * (step + 1),
                task_features=features,
                env_reward=1.0,
                raw_reward=1.0,
                unclipped_reward=1.0,
                clipped_reward=1.0,
                frames_skipped=4,
            )

        storage.compute_returns_and_advantages(
            torch.tensor([0.0]),
            discount_factor=1.0,
            gae_lambda=1.0,
        )
        batches = list(storage.minibatches(2, device="cpu", shuffle=False))

        self.assertTrue(storage.full)
        self.assertEqual(2, len(batches))
        self.assertEqual((2, 4, 84, 84), tuple(batches[0].observation.shape))
        self.assertEqual((2, 1, 8), tuple(batches[0].hidden_state.shape))
        self.assertEqual((2, 5), tuple(batches[0].task_features.shape))
        self.assertTrue(
            np.allclose(np.array([2.5, 1.5, 0.5], dtype=np.float32), storage.advantages[:, 0])
        )
        self.assertTrue(
            np.allclose(np.array([3.0, 2.0, 1.0], dtype=np.float32), storage.returns[:, 0])
        )
        self.assertTrue(np.all(storage.frames_skipped == 4))

    def test_rollout_storage_resets_insert_position(self):
        storage = RolloutStorage(
            rollout_steps=1,
            num_envs=1,
            observation_shape=(4, 84, 84),
            hidden_state_shape=(1, 8),
        )
        storage.insert(
            np.zeros((4, 84, 84), dtype=np.uint8),
            0,
            0.0,
            0.0,
            False,
            False,
            0.0,
            np.zeros((1, 1, 8), dtype=np.float32),
        )

        storage.reset()

        self.assertEqual(0, len(storage))
        self.assertFalse(storage.full)
