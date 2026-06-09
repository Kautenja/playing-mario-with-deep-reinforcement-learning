"""Exploration schedule and seeded action-selection tests."""
from __future__ import annotations

from unittest import TestCase

import torch

from mario_rl.schedules import EpsilonGreedyActionSelector, LinearEpsilonSchedule


class ScheduleTest(TestCase):
    """Validate checkpoint-friendly epsilon behavior."""

    def test_linear_epsilon_schedule_values_are_bounded(self):
        schedule = LinearEpsilonSchedule(start=1.0, final=0.1, decay_frames=10)

        self.assertEqual(1.0, schedule.value(0))
        self.assertAlmostEqual(0.55, schedule.value(5))
        self.assertEqual(0.1, schedule.value(10))
        self.assertEqual(0.1, schedule.value(20))

    def test_schedule_state_round_trips(self):
        schedule = LinearEpsilonSchedule(start=1.0, final=0.1, decay_frames=10)
        schedule.step(3)

        restored = LinearEpsilonSchedule(start=0.0, final=0.0, decay_frames=1)
        restored.load_state_dict(schedule.state_dict())

        self.assertEqual(schedule.state_dict(), restored.state_dict())
        self.assertEqual(schedule.value(), restored.value())

    def test_seeded_random_action_selection_is_reproducible(self):
        q_values = torch.tensor([0.1, 0.9, 0.2])
        first = EpsilonGreedyActionSelector(num_actions=3, seed=123)
        second = EpsilonGreedyActionSelector(num_actions=3, seed=123)

        first_actions = [first.select(q_values, epsilon=1.0) for _ in range(8)]
        second_actions = [second.select(q_values, epsilon=1.0) for _ in range(8)]

        self.assertEqual(first_actions, second_actions)
        self.assertEqual(1, first.select(q_values, epsilon=0.0))
