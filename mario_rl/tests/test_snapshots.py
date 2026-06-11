"""Snapshot curriculum library contract tests."""
from __future__ import annotations

import json
from unittest import TestCase

import numpy as np

from mario_rl.config import SnapshotCurriculumConfig
from mario_rl.snapshots import (
    SnapshotCompatibilityError,
    SnapshotLibrary,
)
from mario_rl.tests.fakes import FakeMarioEnv


class SnapshotLibraryTest(TestCase):
    """Validate process-local capture, sampling, fallback, and errors."""

    def test_capture_sample_and_restore_fake_env_snapshot(self):
        config = SnapshotCurriculumConfig(
            enabled=True,
            max_snapshots=4,
            capture_interval_steps=1,
            sample_probability=1.0,
            tags=("manual",),
        )
        library = SnapshotLibrary(config, seed=7)
        env = FakeMarioEnv(episode_length=5, env_id="FakeMario-v0")

        obs, _ = env.reset(seed=123)
        obs, _, _, _, info = env.step(1)
        entry = library.capture(
            env,
            observation=obs,
            info=info,
            env_id="FakeMario-v0",
            action_set="complex",
            seed_lineage=(123, "episode-0"),
            episode_step=1,
        )
        env.step(1)

        sampled = library.sample_start(
            env,
            env_id="FakeMario-v0",
            action_set="complex",
        )
        self.assertIs(entry, sampled)
        restored_obs, reset_info = library.restore(env, entry, seed=123)
        next_obs, _, _, _, next_info = env.step(1)

        self.assertTrue(reset_info["snapshot_start"])
        self.assertEqual("snapshot-000000", reset_info["snapshot_id"])
        self.assertIn("manual", reset_info["snapshot_tags"])
        self.assertTrue(np.array_equal(obs, restored_obs))
        self.assertEqual(2, next_info["fake_step"])
        self.assertFalse(np.array_equal(restored_obs, next_obs))

        payload = library.payload()
        payload_text = json.dumps(payload, sort_keys=True)
        self.assertEqual(1, payload["counts"]["captured"])
        self.assertEqual(1, payload["counts"]["restored"])
        self.assertIn("compatibility_key", payload["entries"][0])
        self.assertNotIn("native_snapshot", payload_text)
        self.assertNotIn('"observation"', payload_text)
        self.assertFalse(payload["serialization"]["artifact_contains_rom_bytes"])

    def test_reset_or_restore_falls_back_until_compatible_snapshot_exists(self):
        library = SnapshotLibrary(
            SnapshotCurriculumConfig(enabled=True, sample_probability=1.0),
            seed=11,
        )
        env = FakeMarioEnv(episode_length=5, env_id="FakeMario-v0")

        obs, info = library.reset_or_restore(
            env,
            env_id="FakeMario-v0",
            action_set="complex",
            seed=123,
        )
        self.assertFalse(info["snapshot_start"])
        self.assertEqual(1, library.fallback_reset_count)
        self.assertEqual(0, int(obs.max()))

        obs, _, _, _, step_info = env.step(2)
        library.capture(
            env,
            observation=obs,
            info=step_info,
            env_id="FakeMario-v0",
            action_set="complex",
        )
        env.step(2)
        restored, info = library.reset_or_restore(
            env,
            env_id="FakeMario-v0",
            action_set="complex",
            seed=123,
        )

        self.assertTrue(info["snapshot_start"])
        self.assertEqual(1, int(restored.max()))
        self.assertEqual(1, library.restore_count)

    def test_incompatible_snapshot_restore_raises_clear_error_and_sampling_skips(self):
        library = SnapshotLibrary(
            SnapshotCurriculumConfig(enabled=True, sample_probability=1.0),
            seed=13,
        )
        source = FakeMarioEnv(env_id="FakeMario-A-v0")
        target = FakeMarioEnv(env_id="FakeMario-B-v0")

        obs, _ = source.reset(seed=1)
        obs, _, _, _, info = source.step(0)
        entry = library.capture(
            source,
            observation=obs,
            info=info,
            env_id="FakeMario-A-v0",
            action_set="complex",
        )

        self.assertIsNone(
            library.sample_start(
                target,
                env_id="FakeMario-B-v0",
                action_set="complex",
            )
        )
        with self.assertRaisesRegex(
            SnapshotCompatibilityError,
            "snapshot compatibility mismatch",
        ):
            library.restore(target, entry, seed=1)

    def test_maybe_capture_respects_progress_threshold(self):
        library = SnapshotLibrary(
            SnapshotCurriculumConfig(
                enabled=True,
                min_progress=3.0,
                capture_interval_steps=1,
            ),
            seed=17,
        )
        env = FakeMarioEnv(episode_length=5, env_id="FakeMario-v0")
        obs, _ = env.reset(seed=1)
        obs, _, _, _, info = env.step(0)

        self.assertIsNone(
            library.maybe_capture(
                env,
                observation=obs,
                info=info,
                env_id="FakeMario-v0",
                action_set="complex",
                global_step=1,
            )
        )
        self.assertEqual(0, len(library.entries))
