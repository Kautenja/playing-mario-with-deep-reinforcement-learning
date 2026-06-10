"""Evaluation matrix construction and fake-env artifact tests."""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.evaluation_matrix import (
    ConstantPolicy,
    EvaluationMatrixConfig,
    build_evaluation_matrix,
    expand_evaluation_seeds,
    matrix_video_prefix,
    run_evaluation_matrix,
)
from mario_rl.tests.fakes import fake_env_factory, tiny_training_config


class EvaluationMatrixConstructionTest(TestCase):
    """Validate matrix filters and metadata-only catalog reporting."""

    def test_matrix_construction_filters_explicit_lists_and_max_tasks(self):
        matrix = build_evaluation_matrix(
            EvaluationMatrixConfig(
                game_families=("smb1", "smb3"),
                single_stage=True,
                splits=("eval",),
                include_validated=True,
                include_env_ids=(
                    "SuperMarioBros-1-1-v0",
                    "SuperMarioBros3-1-1-v0",
                ),
                max_tasks=1,
            )
        )

        self.assertEqual(1, len(matrix.tasks))
        self.assertEqual("SuperMarioBros-1-1-v0", matrix.tasks[0].env_id)
        self.assertTrue(matrix.tasks[0].runnable)

    def test_smb3_validated_tasks_and_catalog_metadata_are_available(self):
        matrix = build_evaluation_matrix(
            EvaluationMatrixConfig(
                game_families=("smb3",),
                single_stage=True,
                splits=("eval",),
                include_validated=True,
                include_smb3_catalog=True,
            )
        )

        env_ids = {entry.env_id for entry in matrix.tasks}
        self.assertIn("SuperMarioBros3-1-1-v0", env_ids)
        self.assertIn("SuperMarioBros3-1-2-v0", env_ids)
        self.assertIn("SuperMarioBros3-1-4-v0", env_ids)
        self.assertIn("SuperMarioBros3-1-6-v0", env_ids)
        self.assertEqual(56, len(matrix.metadata))
        self.assertTrue(any(not entry.registered for entry in matrix.metadata))
        self.assertTrue(all(not entry.runnable for entry in matrix.metadata))

    def test_empty_matrix_and_invalid_task_ids_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "unknown evaluation matrix env ID"):
            build_evaluation_matrix(
                EvaluationMatrixConfig(include_env_ids=("SuperMarioBros9-9-9-v0",))
            )

        with self.assertRaisesRegex(ValueError, "no candidate tasks"):
            build_evaluation_matrix(
                EvaluationMatrixConfig(game_families=("missing-family",))
            )

    def test_deterministic_seed_expansion(self):
        self.assertEqual(
            (7, 8, 9),
            expand_evaluation_seeds(EvaluationMatrixConfig(seed=7, seed_count=3)),
        )
        self.assertEqual(
            (100, 200),
            expand_evaluation_seeds(EvaluationMatrixConfig(seeds=(100, 200))),
        )

    def test_video_prefix_is_stable_and_task_scoped(self):
        matrix = build_evaluation_matrix(
            EvaluationMatrixConfig(include_env_ids=("SuperMarioBros3-1-1-v0",))
        )
        prefix = matrix_video_prefix(
            EvaluationMatrixConfig(video_name_prefix="matrix"),
            matrix.tasks[0],
            seed=42,
            episode=3,
        )

        self.assertEqual("matrix-supermariobros3-1-1-v0-seed-42-episode-3", prefix)


class EvaluationMatrixArtifactTest(TestCase):
    """Validate no-ROM matrix evaluation with fake policy and fake envs."""

    def test_fake_env_evaluation_writes_json_and_episode_table(self):
        with TemporaryDirectory() as tmpdir:
            config = replace(
                tiny_training_config(tmpdir),
                experiment_name="fake_matrix",
                evaluation_matrix=EvaluationMatrixConfig(
                    include_env_ids=(
                        "SuperMarioBros-1-1-v0",
                        "SuperMarioBros3-1-1-v0",
                    ),
                    seeds=(11, 12),
                    episodes_per_task=2,
                    include_smb3_catalog=True,
                ),
                eval=replace(tiny_training_config(tmpdir).eval, max_steps=3),
            )

            payload = run_evaluation_matrix(
                config,
                env_factory=fake_env_factory,
                policy_factory=lambda _config: ConstantPolicy(action=1),
            )

            self.assertEqual(8, payload["row_count"])
            self.assertEqual(2, payload["matrix"]["task_count"])
            self.assertEqual(56, payload["matrix"]["metadata_count"])
            self.assertEqual(8, payload["global"]["episode_count"])
            summary_path = Path(payload["summary_path"])
            table_path = Path(payload["table_path"])
            self.assertTrue(summary_path.is_file())
            self.assertTrue(table_path.is_file())
            summary = json.loads(summary_path.read_text())
            self.assertEqual(8, summary["row_count"])
            with table_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(8, len(rows))
            self.assertEqual(
                {
                    "SuperMarioBros-1-1-v0",
                    "SuperMarioBros3-1-1-v0",
                },
                {row["env_id"] for row in rows},
            )

    def test_video_config_is_generated_without_real_video_writer(self):
        captured = []

        def capture_env_factory(config):
            captured.append(config.env)
            return fake_env_factory(config)

        with TemporaryDirectory() as tmpdir:
            base = tiny_training_config(tmpdir)
            config = replace(
                base,
                experiment_name="fake_matrix_video",
                env=replace(base.env, render_mode=None),
                evaluation_matrix=EvaluationMatrixConfig(
                    include_env_ids=("SuperMarioBros3-1-1-v0",),
                    seeds=(99,),
                    video_enabled=True,
                    video_name_prefix="matrix-video",
                ),
                eval=replace(base.eval, max_steps=1),
            )

            payload = run_evaluation_matrix(
                config,
                env_factory=capture_env_factory,
                policy_factory=lambda _config: ConstantPolicy(action=0),
            )

        self.assertEqual(1, payload["row_count"])
        self.assertEqual("rgb_array", captured[0].render_mode)
        self.assertTrue(captured[0].video_enabled)
        self.assertIn(
            "matrix-video-supermariobros3-1-1-v0-seed-99-episode-0",
            captured[0].video_name_prefix,
        )
