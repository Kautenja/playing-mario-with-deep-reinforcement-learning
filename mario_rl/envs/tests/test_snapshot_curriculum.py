"""Real Mario snapshot curriculum smoke tests."""
from __future__ import annotations

import json
from unittest import TestCase

import numpy as np

from mario_rl.config import SnapshotCurriculumConfig
from mario_rl.envs import make_env
from mario_rl.snapshots import SnapshotLibrary


class MarioSnapshotCurriculumSmokeTest(TestCase):
    """Exercise process-local snapshots through the public env factory."""

    def test_real_mario_snapshot_can_restore_preprocessed_pixel_start(self):
        env_id = "SuperMarioBros-1-1-v0"
        env = make_env(
            env_id,
            render_mode="rgb_array",
            seed=123,
            action_set="right_only",
            frame_skip=2,
            frame_stack=2,
            image_size=(16, 16),
            record_statistics=False,
            max_episode_steps=None,
            no_progress_timeout_steps=None,
        )
        library = SnapshotLibrary(
            SnapshotCurriculumConfig(
                enabled=True,
                capture_interval_steps=1,
                sample_probability=1.0,
                tags=("real-smoke",),
            ),
            seed=123,
        )

        try:
            env.reset(seed=123)
            snapshot_obs, _, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(2, info["frames_skipped"])
            entry = library.capture(
                env,
                observation=snapshot_obs,
                info=info,
                env_id=env_id,
                action_set=env.mario_rl_action_set,
                seed_lineage=(123,),
                episode_step=1,
            )
            env.step(1)
            restored_obs, reset_info = library.restore(env, entry, seed=123)
            next_obs, _, _, _, next_info = env.step(1)
        finally:
            env.close()

        self.assertTrue(reset_info["snapshot_start"])
        self.assertEqual(entry.metadata.snapshot_id, reset_info["snapshot_id"])
        self.assertEqual((2, 16, 16), restored_obs.shape)
        self.assertTrue(np.array_equal(snapshot_obs, restored_obs))
        self.assertEqual((2, 16, 16), next_obs.shape)
        self.assertIn("task_id", next_info)

        payload_text = json.dumps(library.payload(), sort_keys=True)
        self.assertIn("rom_sha256", payload_text)
        self.assertNotIn("super-mario-bros.nes", payload_text)
        self.assertNotIn("native_snapshot", payload_text)
        self.assertNotIn('"observation"', payload_text)
