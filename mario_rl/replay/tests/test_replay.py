"""Replay buffer contract tests."""
from __future__ import annotations

from dataclasses import replace
from unittest import TestCase

import numpy as np
import torch

from mario_rl.config import MarioRLConfig
from mario_rl.replay import UniformReplayBuffer, build_replay_buffer


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

    def test_empty_sample_is_rejected(self):
        replay = UniformReplayBuffer(capacity=2, state_shape=(4, 8, 8), seed=123)

        with self.assertRaises(ValueError):
            replay.sample(1)

    def test_replay_factory_uses_config_and_gates_prioritized_replay(self):
        config = MarioRLConfig()
        replay = build_replay_buffer(config, seed=123)

        self.assertIsInstance(replay, UniformReplayBuffer)
        self.assertEqual(config.replay.capacity, replay.capacity)
        self.assertEqual(config.replay.state_shape, replay.state_shape)

        prioritized = replace(config, replay=replace(config.replay, prioritized=True))
        with self.assertRaises(NotImplementedError):
            build_replay_buffer(prioritized)
