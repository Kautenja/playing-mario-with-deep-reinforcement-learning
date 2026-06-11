"""Structured Mario metrics accumulator tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.metrics import (
    MarioMetricsAccumulator,
    StepMetrics,
    write_metrics_json,
    write_summary_csv,
)


class MarioMetricsAccumulatorTest(TestCase):
    """Validate step, episode, and grouped Mario metrics."""

    def test_accumulates_global_family_and_task_summaries(self):
        metrics = MarioMetricsAccumulator(default_task_id="SuperMarioBros-1-1-v0")
        metrics.start_episode(fallback_env_id="SuperMarioBros-1-1-v0")
        metrics.observe_step(
            reward=1.0,
            transformed_reward=2.0,
            terminated=True,
            info={
                "task_id": "SuperMarioBros-1-1-v0",
                "game_family": "smb1",
                "world": 1,
                "stage": 1,
                "progress": 12,
                "progress_max": 16,
                "clear": True,
                "death": False,
                "timeout": False,
                "reward_total_unclipped": 3.0,
                "reward_total_clipped": 1.0,
                "reward_components": {"progress": 3.0, "death": 0.0},
            },
        )
        metrics.finish_episode(terminated=True, truncated=False)

        metrics.start_episode(fallback_env_id="SuperMarioBros3-1-1-v0")
        metrics.observe_step(
            reward=-1.0,
            transformed_reward=-0.5,
            truncated=True,
            info={
                "task_id": "SuperMarioBros3-1-1-v0",
                "game_family": "smb3",
                "world": 1,
                "stage": 1,
                "progress": 4,
                "progress_max": 8,
                "clear": False,
                "death": True,
                "timeout": True,
                "reward_components": {"progress": -1.0, "death": -5.0},
            },
        )
        metrics.finish_episode(terminated=False, truncated=True)

        payload = metrics.to_payload()
        global_metrics = payload["global"]
        self.assertEqual(2, global_metrics["episode_count"])
        self.assertEqual(1, global_metrics["clear_count"])
        self.assertEqual(0.5, global_metrics["clear_rate"])
        self.assertEqual(1, global_metrics["death_count"])
        self.assertEqual(1, global_metrics["timeout_count"])
        self.assertEqual(1, global_metrics["truncation_count"])
        self.assertEqual(16.0, global_metrics["max_progress"])
        self.assertEqual(8.0, global_metrics["final_progress_mean"])
        self.assertEqual(0, global_metrics["snapshot_start_count"])
        self.assertEqual(2, global_metrics["full_reset_episode_count"])
        self.assertEqual(1, global_metrics["full_reset_clear_count"])
        self.assertEqual(2.0, global_metrics["reward_component_sums"]["progress"])
        self.assertIn("smb1", payload["by_game_family"])
        self.assertIn("smb3", payload["by_game_family"])
        self.assertIn("SuperMarioBros-1-1-v0", payload["by_task"])

    def test_missing_optional_keys_are_counted_and_fallback_task_is_used(self):
        metrics = MarioMetricsAccumulator(default_task_id="FakeMario-v0")
        metrics.start_episode()
        step = metrics.observe_step(reward=1.0, info={})
        metrics.finish_episode(terminated=False, truncated=True)

        self.assertIsInstance(step, StepMetrics)
        payload = metrics.to_payload()
        episode = payload["episodes"][0]
        self.assertEqual("FakeMario-v0", episode["task_id"])
        self.assertEqual("unknown", episode["game_family"])
        self.assertEqual(1, episode["missing_info_counts"]["clear"])
        self.assertEqual(1, episode["missing_info_counts"]["task_id"])
        self.assertEqual(1, payload["global"]["missing_info_counts"]["progress"])

    def test_reward_component_sums_and_json_csv_serialization(self):
        metrics = MarioMetricsAccumulator(default_task_id="SuperMarioBros-1-1-v0")
        metrics.start_episode()
        metrics.observe_step(
            reward=2.0,
            transformed_reward=4.0,
            info={
                "task_id": "SuperMarioBros-1-1-v0",
                "game_family": "smb1",
                "world": 1,
                "stage": 1,
                "progress": 5,
                "progress_max": 6,
                "reward_components": {"progress": 2.0, "coins": 1.5},
            },
        )
        metrics.observe_step(
            reward=3.0,
            transformed_reward=6.0,
            info={
                "task_id": "SuperMarioBros-1-1-v0",
                "game_family": "smb1",
                "world": 1,
                "stage": 1,
                "progress": 7,
                "progress_max": 9,
                "reward_components": {"progress": 3.0, "coins": 0.5},
            },
        )
        metrics.finish_episode(terminated=True, truncated=False)
        payload = metrics.to_payload()

        with TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "metrics.json"
            csv_path = Path(tmpdir) / "metrics.csv"
            write_metrics_json(json_path, payload)
            write_summary_csv(csv_path, payload)
            self.assertEqual(5.0, json.loads(json_path.read_text())["global"]["episode_return_total"])
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual("10.0", rows[0]["transformed_return_total"])
            self.assertIn('"coins": 2.0', rows[0]["reward_component_sums_json"])
