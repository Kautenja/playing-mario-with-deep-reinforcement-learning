"""Replay buffer contract tests."""
from __future__ import annotations

from dataclasses import replace
from unittest import TestCase

import numpy as np
import torch

from mario_rl.config import MarioRLConfig
from mario_rl.envs import TaskFeatureEncoder
from mario_rl.replay import (
    PrioritizedReplayBuffer,
    UniformReplayBuffer,
    build_replay_buffer,
)


class ReplayBufferTest(TestCase):
    """Validate uniform replay shapes, dtypes, and config behavior."""

    def _state(self, value):
        return np.full((4, 8, 8), value, dtype=np.uint8)

    def test_uniform_replay_samples_partial_buffer_without_none_entries(self):
        replay = UniformReplayBuffer(capacity=8, state_shape=(4, 8, 8), seed=123)
        for index in range(3):
            replay.push(
                self._state(index),
                action=index,
                reward=float(index),
                terminated=index == 1,
                truncated=index == 2,
                next_state=self._state(index + 1),
            )

        batch = replay.sample(5)

        self.assertEqual((5, 4, 8, 8), batch.state.shape)
        self.assertEqual((5,), batch.action.shape)
        self.assertEqual((5,), batch.reward.shape)
        self.assertEqual((5,), batch.terminated.shape)
        self.assertEqual((5,), batch.truncated.shape)
        self.assertEqual((5, 4, 8, 8), batch.next_state.shape)
        self.assertEqual(np.uint8, batch.state.dtype)
        self.assertEqual(np.int64, batch.action.dtype)
        self.assertEqual(np.float32, batch.reward.dtype)
        self.assertEqual(np.bool_, batch.terminated.dtype)
        self.assertEqual(np.bool_, batch.truncated.dtype)

    def test_reward_info_survives_push_sample_and_to_torch(self):
        replay = UniformReplayBuffer(
            capacity=2,
            state_shape=(4, 8, 8),
            store_reward_info=True,
            seed=123,
        )
        replay.push(
            self._state(1),
            2,
            1.25,
            False,
            True,
            self._state(2),
            env_reward=3.0,
            raw_reward=4.0,
            unclipped_reward=5.0,
            clipped_reward=6.0,
        )

        batch = replay.sample(1)
        torch_batch = batch.to_torch(device=torch.device("cpu"))

        self.assertEqual(1.25, float(batch.reward[0]))
        self.assertEqual(3.0, float(batch.env_reward[0]))
        self.assertEqual(4.0, float(batch.raw_reward[0]))
        self.assertEqual(5.0, float(batch.unclipped_reward[0]))
        self.assertEqual(6.0, float(batch.clipped_reward[0]))
        self.assertEqual(torch.float32, torch_batch.raw_reward.dtype)
        self.assertEqual(5.0, float(torch_batch.unclipped_reward[0]))

    def test_replay_batch_can_move_to_torch_device(self):
        replay = UniformReplayBuffer(capacity=2, state_shape=(4, 8, 8), seed=123)
        replay.push(self._state(1), 2, 1.5, False, True, self._state(2))

        batch = replay.sample(1, device=torch.device("cpu"))

        self.assertEqual(torch.device("cpu"), batch.state.device)
        self.assertEqual(torch.uint8, batch.state.dtype)
        self.assertEqual(torch.long, batch.action.dtype)
        self.assertEqual(torch.float32, batch.reward.dtype)
        self.assertEqual(torch.bool, batch.terminated.dtype)
        self.assertEqual(torch.bool, batch.truncated.dtype)
        self.assertIsNone(batch.indices)
        self.assertIsNone(batch.importance_weights)

    def test_task_features_survive_push_sample_and_to_torch(self):
        encoder = TaskFeatureEncoder()
        features = encoder.encode_env_id("SuperMarioBros-1-1-v0").vector
        replay = UniformReplayBuffer(
            capacity=2,
            state_shape=(4, 8, 8),
            task_feature_shape=(encoder.feature_size,),
            seed=123,
        )

        replay.push(
            self._state(1),
            2,
            1.5,
            False,
            True,
            self._state(2),
            task_features=features,
            next_task_features=features,
        )
        batch = replay.sample(1)
        torch_batch = batch.to_torch(device=torch.device("cpu"))

        self.assertEqual((1, encoder.feature_size), batch.task_features.shape)
        self.assertTrue(np.array_equal(features, batch.task_features[0]))
        self.assertEqual(torch.float32, torch_batch.task_features.dtype)
        self.assertEqual((1, encoder.feature_size), tuple(torch_batch.task_features.shape))

    def test_empty_sample_is_rejected(self):
        replay = UniformReplayBuffer(capacity=2, state_shape=(4, 8, 8), seed=123)

        with self.assertRaises(ValueError):
            replay.sample(1)

    def test_prioritized_replay_samples_high_priority_items_more_often(self):
        replay = PrioritizedReplayBuffer(
            capacity=4,
            state_shape=(4, 8, 8),
            priority_alpha=1.0,
            priority_beta=0.5,
            seed=123,
        )
        for index in range(4):
            replay.push(
                self._state(index),
                action=index,
                reward=float(index),
                terminated=False,
                truncated=False,
                next_state=self._state(index + 1),
            )
        replay.update_priorities(
            np.arange(4),
            np.asarray([1.0, 1.0, 1.0, 100.0], dtype=np.float32),
        )

        counts = np.zeros(4, dtype=np.int64)
        for _ in range(500):
            batch = replay.sample(1)
            self.assertIsNotNone(batch.indices)
            self.assertIsNotNone(batch.importance_weights)
            counts[int(batch.indices[0])] += 1

        self.assertGreater(counts[3], 400)
        self.assertLess(counts[:3].max(), counts[3])

    def test_prioritized_replay_updates_are_deterministic_with_fixed_seed(self):
        first = PrioritizedReplayBuffer(capacity=5, state_shape=(4, 8, 8), seed=99)
        second = PrioritizedReplayBuffer(capacity=5, state_shape=(4, 8, 8), seed=99)
        for replay in (first, second):
            for index in range(5):
                replay.push(
                    self._state(index),
                    action=index,
                    reward=float(index),
                    terminated=False,
                    truncated=False,
                    next_state=self._state(index + 1),
                )
            replay.update_priorities(
                np.arange(5),
                np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            )

        first_batch = first.sample(4)
        second_batch = second.sample(4)
        self.assertTrue(np.array_equal(first_batch.indices, second_batch.indices))
        self.assertTrue(
            np.allclose(
                first_batch.importance_weights,
                second_batch.importance_weights,
            )
        )

        new_priorities = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        first.update_priorities(first_batch.indices, new_priorities)
        second.update_priorities(second_batch.indices, new_priorities)

        self.assertEqual(first.priority_summary(), second.priority_summary())
        self.assertTrue(
            np.array_equal(first.sample(4).indices, second.sample(4).indices)
        )

    def test_replay_factory_uses_config_and_builds_prioritized_replay(self):
        config = MarioRLConfig()
        replay = build_replay_buffer(config, seed=123)

        self.assertIsInstance(replay, UniformReplayBuffer)
        self.assertEqual(config.replay.capacity, replay.capacity)
        self.assertEqual(config.replay.state_shape, replay.state_shape)
        self.assertTrue(replay.store_reward_info)

        prioritized = replace(config, replay=replace(config.replay, prioritized=True))
        prioritized_replay = build_replay_buffer(prioritized, seed=123)
        self.assertIsInstance(prioritized_replay, PrioritizedReplayBuffer)
        self.assertEqual(
            config.replay.priority_alpha,
            prioritized_replay.priority_alpha,
        )
        self.assertEqual(
            config.replay.priority_beta,
            prioritized_replay.priority_beta,
        )

        conditioned = replace(
            config,
            model=replace(config.model, task_conditioning=True, task_feature_size=0),
        )
        conditioned_replay = build_replay_buffer(conditioned, seed=123)
        self.assertEqual(
            (TaskFeatureEncoder().feature_size,),
            conditioned_replay.task_feature_shape,
        )
