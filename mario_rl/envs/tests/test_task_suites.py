"""Task-suite resolver and sampler contract tests."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from unittest import TestCase

from mario_rl.envs import MarioTask, TaskSuite, TaskSuiteConfig


class TaskSuiteTest(TestCase):
    """Validate deterministic metadata-only task-suite sampling."""

    def test_resolves_single_stage_candidates_across_registered_families(self):
        suite = TaskSuite(
            TaskSuiteConfig(
                game_families=("smb1", "lost_levels", "smb2_usa", "smb3"),
                single_stage=True,
                include_validated=True,
            )
        )

        self.assertEqual(
            {"smb1", "lost_levels", "smb2_usa", "smb3"},
            set(suite.family_counts),
        )
        self.assertIn("SuperMarioBros-1-1-v0", suite.env_ids)
        self.assertIn("SuperMarioBros2-1-1-v0", suite.env_ids)
        self.assertIn("SuperMarioBros2USA-1-1-v0", suite.env_ids)
        self.assertIn("SuperMarioBros3-1-1-v0", suite.env_ids)
        self.assertNotIn("SuperMarioBrosRandomStages-v0", suite.env_ids)

    def test_resolves_full_game_candidates_without_stage_ids(self):
        suite = TaskSuite(
            TaskSuiteConfig(
                game_families=("smb1", "lost_levels", "smb2_usa", "smb3"),
                single_stage=False,
                include_validated=True,
            )
        )

        self.assertIn("SuperMarioBros-v0", suite.env_ids)
        self.assertIn("SuperMarioBros2-v0", suite.env_ids)
        self.assertIn("SuperMarioBros2USA-v0", suite.env_ids)
        self.assertIn("SuperMarioBros3-v0", suite.env_ids)
        self.assertTrue(all(not task.single_stage for task in suite.candidates))

    def test_seeded_sampling_is_deterministic_and_seed_sensitive(self):
        config = TaskSuiteConfig(
            include_env_ids=("SuperMarioBros-1-1-v0", "SuperMarioBros3-1-1-v0"),
            single_stage=True,
            seed=17,
        )
        first = [TaskSuite(config).task_for_index(index).env_id for index in range(20)]
        second = [TaskSuite(config).task_for_index(index).env_id for index in range(20)]
        changed_seed = replace(config, seed=18)
        different = [
            TaskSuite(changed_seed).task_for_index(index).env_id
            for index in range(20)
        ]

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_family_weighting_samples_families_before_tasks(self):
        tasks = (
            *(
                MarioTask(
                    env_id=f"LargeFamily-{index}-v0",
                    game="large",
                    game_family="large",
                    version=0,
                    rom_mode="test",
                    world=index + 1,
                    stage=1,
                    single_stage=True,
                )
                for index in range(80)
            ),
            MarioTask(
                env_id="SmallFamily-1-v0",
                game="small",
                game_family="small",
                version=0,
                rom_mode="test",
                world=1,
                stage=1,
                single_stage=True,
            ),
        )
        suite = TaskSuite(
            TaskSuiteConfig(
                family_weights={"large": 1.0, "small": 1.0},
                seed=99,
            ),
            tasks=tasks,
        )

        counts = Counter(suite.task_for_index(index).game_family for index in range(1000))

        self.assertGreater(counts["small"], 400)
        self.assertLess(counts["small"], 600)

    def test_invalid_filters_and_empty_candidates_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "splits"):
            TaskSuite(TaskSuiteConfig(splits=("holdout",)))

        with self.assertRaisesRegex(ValueError, "no candidate"):
            TaskSuite(TaskSuiteConfig(game_families=("missing",)))

        with self.assertRaisesRegex(ValueError, "family_weights"):
            TaskSuite(TaskSuiteConfig(family_weights={"missing": 1.0}))

    def test_smb3_catalog_reports_unregistered_matrix_without_sampling_it(self):
        suite = TaskSuite(TaskSuiteConfig(game_families=("smb3",), single_stage=True))

        registered = set(suite.env_ids)
        full_catalog = suite.smb3_catalog(validated=None)
        unvalidated = tuple(stage for stage in full_catalog if not stage.validated)

        self.assertGreater(len(full_catalog), len(registered))
        self.assertTrue(unvalidated)
        self.assertFalse(any(stage.env_id in registered for stage in unvalidated))
