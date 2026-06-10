"""Task feature encoder contract tests."""
from __future__ import annotations

from unittest import TestCase

import numpy as np

from mario_rl.envs import (
    TaskFeatureEncoder,
    UNKNOWN_TASK_VALUE,
    available_tasks,
    smb3_stage_matrix,
)


class TaskFeatureEncoderTest(TestCase):
    """Validate deterministic task conditioning features without env creation."""

    def test_encoder_covers_registered_game_families(self):
        encoder = TaskFeatureEncoder()
        families = {task.game_family for task in available_tasks()}

        self.assertEqual({"smb1", "lost_levels", "smb2_usa", "smb3"}, families)
        for family in sorted(families):
            with self.subTest(game_family=family):
                task = next(task for task in available_tasks(game_family=family))
                features = encoder.encode_task(task)
                self.assertEqual(family, features.game_family)
                self.assertEqual((encoder.feature_size,), features.vector.shape)
                self.assertEqual(np.float32, features.vector.dtype)

    def test_alias_ids_encode_to_canonical_task_features(self):
        encoder = TaskFeatureEncoder()

        canonical = encoder.encode_env_id("SuperMarioBros-1-1-v0")
        alias = encoder.encode_env_id("SuperMarioBros1-1-v0")

        self.assertEqual("SuperMarioBros-1-1-v0", alias.task_id)
        self.assertTrue(np.array_equal(canonical.vector, alias.vector))

    def test_unknown_env_id_uses_explicit_unknown_values(self):
        encoder = TaskFeatureEncoder()

        features = encoder.encode_env_id("CustomMario-v0")

        self.assertEqual("CustomMario-v0", features.env_id)
        self.assertEqual(UNKNOWN_TASK_VALUE, features.task_id)
        self.assertEqual(UNKNOWN_TASK_VALUE, features.game_family)
        self.assertEqual(UNKNOWN_TASK_VALUE, features.rom_mode)
        self.assertEqual((encoder.feature_size,), features.vector.shape)
        self.assertEqual(1.0, features.vector[0])
        self.assertFalse(features.single_stage)
        self.assertFalse(features.validated)

    def test_smb3_validated_stage_matrix_entries_are_supported(self):
        encoder = TaskFeatureEncoder()
        stages = smb3_stage_matrix(validated=True)

        self.assertGreater(len(stages), 0)
        for stage in stages:
            with self.subTest(env_id=stage.env_id):
                features = encoder.encode_env_id(stage.env_id)
                self.assertEqual("smb3", features.game_family)
                self.assertEqual(stage.world, features.world)
                self.assertEqual(stage.stage, features.stage)
                self.assertTrue(features.single_stage)
                self.assertTrue(features.validated)
