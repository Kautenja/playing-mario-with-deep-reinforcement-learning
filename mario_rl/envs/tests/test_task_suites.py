"""Task-suite resolver and sampler contract tests."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mario_rl.envs import AdaptiveCurriculum, MarioTask, TaskSuite, TaskSuiteConfig
from mario_rl.metrics import EpisodeMetrics, TaskMetricKey


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

    def test_adaptive_curriculum_unlocks_frontier_in_order(self):
        tasks = _curriculum_tasks()
        curriculum = AdaptiveCurriculum(
            TaskSuiteConfig(
                mode="adaptive",
                seed=11,
                curriculum_mastery_min_episodes=1,
                curriculum_mastery_clear_rate=1.0,
                curriculum_mastery_death_rate=0.0,
            ),
            tasks=tasks,
        )

        first = curriculum.task_for_episode(0)
        self.assertEqual("SuperMarioBros-1-1-v0", first.env_id)
        self.assertEqual("SuperMarioBros-1-1-v0", curriculum.task_for_episode(1).env_id)
        self.assertIn(
            "SuperMarioBros2-1-1-v0",
            curriculum.metadata()["locked_env_ids"],
        )

        curriculum.observe_episode(
            _episode(first.env_id, clear=True, progress=150.0),
            env_id=first.env_id,
        )
        second = curriculum.task_for_episode(2)
        self.assertEqual("SuperMarioBros-1-2-v0", second.env_id)
        self.assertIn(
            "SuperMarioBros2-1-1-v0",
            curriculum.metadata()["locked_env_ids"],
        )

        curriculum.observe_episode(
            _episode(second.env_id, clear=True, progress=175.0),
            env_id=second.env_id,
        )
        self.assertEqual("SuperMarioBros2-1-1-v0", curriculum.task_for_episode(3).env_id)
        counts = curriculum.summary_counts()
        self.assertEqual(1, counts["active"])
        self.assertEqual(2, counts["mastered"])
        self.assertEqual(2, counts["retired"])

    def test_adaptive_curriculum_sampling_is_seed_deterministic(self):
        config = TaskSuiteConfig(
            mode="adaptive",
            seed=23,
            curriculum_frontier_size=2,
        )

        first = [
            AdaptiveCurriculum(config, tasks=_curriculum_tasks()).task_for_index(index).env_id
            for index in range(8)
        ]
        second = [
            AdaptiveCurriculum(config, tasks=_curriculum_tasks()).task_for_index(index).env_id
            for index in range(8)
        ]
        changed_seed = replace(config, seed=24)
        different = [
            AdaptiveCurriculum(changed_seed, tasks=_curriculum_tasks())
            .task_for_index(index)
            .env_id
            for index in range(8)
        ]

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_adaptive_curriculum_resumes_from_saved_state_artifact(self):
        with TemporaryDirectory() as tmpdir:
            tasks = _curriculum_tasks()
            config = TaskSuiteConfig(
                mode="adaptive",
                seed=5,
                curriculum_mastery_min_episodes=1,
                curriculum_mastery_clear_rate=1.0,
                curriculum_mastery_death_rate=0.0,
            )
            curriculum = AdaptiveCurriculum(config, tasks=tasks)
            first = curriculum.task_for_episode(0)
            curriculum.observe_episode(
                _episode(first.env_id, clear=True, progress=120.0),
                env_id=first.env_id,
            )
            state_path = Path(tmpdir) / "curriculum-state.json"
            state_path.write_text(
                json.dumps({"curriculum": curriculum.payload()}),
                encoding="utf-8",
            )

            resumed = AdaptiveCurriculum(
                replace(config, curriculum_state_path=str(state_path)),
                tasks=tasks,
            )

            records = {record.env_id: record for record in resumed.records}
            self.assertTrue(records["SuperMarioBros-1-1-v0"].mastered)
            self.assertEqual("SuperMarioBros-1-2-v0", resumed.task_for_episode(1).env_id)


def _curriculum_tasks():
    return (
        MarioTask(
            env_id="SuperMarioBros-1-2-v0",
            game="smb1",
            game_family="smb1",
            version=0,
            rom_mode="vanilla",
            world=1,
            stage=2,
            single_stage=True,
        ),
        MarioTask(
            env_id="SuperMarioBros2-1-1-v0",
            game="lost",
            game_family="lost_levels",
            version=0,
            rom_mode="vanilla",
            world=1,
            stage=1,
            single_stage=True,
        ),
        MarioTask(
            env_id="SuperMarioBros-1-1-v0",
            game="smb1",
            game_family="smb1",
            version=0,
            rom_mode="vanilla",
            world=1,
            stage=1,
            single_stage=True,
        ),
    )


def _episode(env_id: str, *, clear: bool, progress: float) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode=0,
        complete=True,
        snapshot_start=False,
        task=TaskMetricKey(task_id=env_id, game_family="smb1"),
        step_count=1,
        frame_count=1,
        episode_return=1.0,
        transformed_return=1.0,
        raw_return=1.0,
        unclipped_return=1.0,
        clipped_return=1.0,
        clear=clear,
        death=False,
        timeout=False,
        terminated=True,
        truncated=False,
        max_progress=progress,
        final_progress=progress,
    )
