"""Auxiliary target extraction contract tests."""
from __future__ import annotations

from unittest import TestCase

from mario_rl.auxiliary import (
    GAME_FAMILY_CLASSES,
    AuxiliaryLossConfig,
    auxiliary_loss_weights,
    auxiliary_output_sizes,
    auxiliary_target_names,
    extract_auxiliary_targets,
)


class AuxiliaryTargetExtractionTest(TestCase):
    """Validate normalized info extraction and missing-target masks."""

    def test_extracts_representative_auxiliary_targets(self):
        info = {
            "clear": True,
            "death": False,
            "game_family": "smb3",
            "progress": 40,
            "progress_max": 80,
            "reward_components": {"progress": 3.0, "death": 0.0},
            "reward_total_unclipped": 12.0,
            "reward_total_clipped": 15.0,
        }

        record = extract_auxiliary_targets(
            info,
            transformed_reward=1.5,
            targets=(
                "progress_delta",
                "progress_normalized",
                "clear",
                "death",
                "transformed_reward",
                "reward_total_unclipped",
                "reward_total_clipped",
                "game_family",
            ),
        )

        self.assertEqual(3.0, record.values["progress_delta"])
        self.assertEqual(0.5, record.values["progress_normalized"])
        self.assertEqual(1.0, record.values["clear"])
        self.assertEqual(0.0, record.values["death"])
        self.assertEqual(1.5, record.values["transformed_reward"])
        self.assertEqual(12.0, record.values["reward_total_unclipped"])
        self.assertEqual(15.0, record.values["reward_total_clipped"])
        self.assertEqual(
            GAME_FAMILY_CLASSES.index("smb3"),
            int(record.values["game_family"]),
        )
        self.assertTrue(all(record.masks.values()))

    def test_missing_targets_are_masked_without_garbage_values(self):
        record = extract_auxiliary_targets(
            {"reward_components": {"death": -25.0}},
            targets=("progress_delta", "clear", "game_family", "transformed_reward"),
        )

        self.assertEqual(0.0, record.values["progress_delta"])
        self.assertFalse(record.masks["progress_delta"])
        self.assertFalse(record.masks["clear"])
        self.assertFalse(record.masks["game_family"])
        self.assertFalse(record.masks["transformed_reward"])


class AuxiliaryLossConfigTest(TestCase):
    """Validate optional auxiliary config expansion and weights."""

    def test_disabled_config_has_no_enabled_targets(self):
        config = AuxiliaryLossConfig()

        self.assertFalse(config.enabled)
        self.assertEqual((), auxiliary_target_names(config))
        self.assertEqual({}, auxiliary_output_sizes(config))

    def test_enabled_config_expands_defaults_and_weights(self):
        config = AuxiliaryLossConfig(
            enabled=True,
            weights={"clear": 0.25, "game_family": 2.0},
        )

        self.assertIn("progress_delta", auxiliary_target_names(config))
        self.assertEqual(1, auxiliary_output_sizes(config)["clear"])
        self.assertEqual(
            len(GAME_FAMILY_CLASSES),
            auxiliary_output_sizes(config)["game_family"],
        )
        weights = auxiliary_loss_weights(config)
        self.assertEqual(0.25, weights["clear"])
        self.assertEqual(2.0, weights["game_family"])
        self.assertEqual(1.0, weights["death"])

    def test_unknown_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown auxiliary target"):
            AuxiliaryLossConfig(enabled=True, targets=("not_a_target",))
