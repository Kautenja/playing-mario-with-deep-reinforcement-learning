"""PyTorch DQN loss and target contract tests."""
from __future__ import annotations

from unittest import TestCase

import torch

from mario_rl.models import (
    DQN,
    compute_auxiliary_loss,
    compute_dqn_loss,
    compute_ppo_loss,
    compute_td_targets,
    gather_action_q_values,
)


class DQNLossTest(TestCase):
    """Validate selected-action TD update helpers."""

    def test_selected_q_values_are_gathered_by_action_index(self):
        q_values = torch.tensor([[0.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
        actions = torch.tensor([2, 0])

        selected = gather_action_q_values(q_values, actions)

        self.assertTrue(torch.equal(torch.tensor([4.0, 3.0]), selected))

    def test_huber_td_loss_matches_fixed_tensor_values(self):
        q_values = torch.tensor([[0.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
        actions = torch.tensor([1, 0])
        targets = torch.tensor([0.0, 4.0])

        loss = compute_dqn_loss(q_values, actions, targets)

        self.assertTrue(torch.allclose(torch.tensor(1.0), loss))

    def test_huber_td_loss_applies_importance_sampling_weights(self):
        q_values = torch.tensor([[0.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
        actions = torch.tensor([1, 0])
        targets = torch.tensor([0.0, 4.0])
        weights = torch.tensor([0.5, 1.0])

        loss = compute_dqn_loss(q_values, actions, targets, sample_weights=weights)

        self.assertTrue(torch.allclose(torch.tensor(0.625), loss))

    def test_td_targets_handle_terminated_and_truncated_transitions(self):
        rewards = torch.tensor([1.0, 1.0, 1.0])
        terminated = torch.tensor([False, True, False])
        truncated = torch.tensor([False, False, True])
        target_next_q = torch.tensor([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])

        targets = compute_td_targets(
            rewards,
            terminated,
            truncated,
            target_next_q,
            discount_factor=0.5,
        )

        self.assertTrue(torch.allclose(torch.tensor([3.5, 1.0, 1.0]), targets))

    def test_double_dqn_target_selection_uses_online_argmax_and_target_values(self):
        rewards = torch.tensor([1.0, 2.0])
        terminated = torch.tensor([False, False])
        truncated = torch.tensor([False, False])
        online_next_q = torch.tensor([[9.0, 1.0], [1.0, 9.0]])
        target_next_q = torch.tensor([[3.0, 30.0], [4.0, 40.0]])

        targets = compute_td_targets(
            rewards,
            terminated,
            truncated,
            target_next_q,
            discount_factor=0.25,
            online_next_q_values=online_next_q,
            double_dqn=True,
        )

        self.assertTrue(torch.allclose(torch.tensor([1.75, 12.0]), targets))

    def test_one_optimizer_step_produces_finite_loss_and_updates_parameters(self):
        torch.manual_seed(123)
        model = DQN(input_channels=4, num_actions=3, input_shape=(4, 84, 84), hidden_size=64)
        target_model = DQN(input_channels=4, num_actions=3, input_shape=(4, 84, 84), hidden_size=64)
        target_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        states = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
        next_states = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
        actions = torch.tensor([0, 1, 2, 1])
        rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
        terminated = torch.tensor([False, False, True, False])
        truncated = torch.tensor([False, True, False, False])

        before = [parameter.detach().clone() for parameter in model.parameters()]
        with torch.no_grad():
            targets = compute_td_targets(
                rewards,
                terminated,
                truncated,
                target_model(next_states),
                discount_factor=0.99,
            )

        loss = compute_dqn_loss(model(states), actions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        )


class PPOLossTest(TestCase):
    """Validate clipped PPO objective helpers."""

    def test_clipped_ppo_loss_matches_fixed_tensor_values(self):
        logits = torch.zeros(2, 2)
        actions = torch.tensor([0, 1])
        old_log_probabilities = torch.log(torch.tensor([0.5, 0.5]))
        values = torch.tensor([0.25, -0.25])
        returns = torch.tensor([1.25, -1.25])
        advantages = torch.tensor([1.0, -1.0])

        loss = compute_ppo_loss(
            logits,
            values,
            actions,
            old_log_probabilities,
            returns,
            advantages,
            value_loss_coefficient=0.5,
            entropy_coefficient=0.01,
            normalize_advantages=False,
        )

        expected_entropy = torch.log(torch.tensor(2.0))
        expected_total = torch.tensor(0.5) - 0.01 * expected_entropy
        self.assertTrue(torch.allclose(torch.tensor(0.0), loss.policy))
        self.assertTrue(torch.allclose(torch.tensor(1.0), loss.value))
        self.assertTrue(torch.allclose(expected_entropy, loss.entropy))
        self.assertTrue(torch.allclose(expected_total, loss.total))
        self.assertTrue(torch.isfinite(loss.approximate_kl))
        self.assertTrue(torch.isfinite(loss.clip_fraction))


class AuxiliaryLossTest(TestCase):
    """Validate masked auxiliary losses and weighted composition."""

    def test_weighted_auxiliary_loss_uses_masks(self):
        predictions = {
            "progress_delta": torch.tensor([0.0, 2.0, 10.0]),
            "clear": torch.tensor([-20.0, 0.0, 0.0]),
        }
        targets = {
            "progress_delta": torch.tensor([1.0, 2.0, 100.0]),
            "clear": torch.tensor([0.0, 1.0, 1.0]),
        }
        masks = {
            "progress_delta": torch.tensor([True, True, False]),
            "clear": torch.tensor([True, False, False]),
        }

        loss = compute_auxiliary_loss(
            predictions,
            targets,
            masks,
            weights={"progress_delta": 2.0, "clear": 0.5},
        )

        self.assertTrue(torch.allclose(torch.tensor(0.25), loss.terms["progress_delta"]))
        self.assertTrue(torch.allclose(torch.tensor(0.0), loss.terms["clear"]))
        self.assertTrue(torch.allclose(torch.tensor(0.5), loss.weighted_terms["progress_delta"]))
        self.assertTrue(torch.allclose(torch.tensor(0.5), loss.total))
        self.assertEqual(2.0, float(loss.valid_counts["progress_delta"]))
        self.assertEqual(1.0, float(loss.valid_counts["clear"]))

    def test_classification_auxiliary_loss_handles_partial_multi_game_batch(self):
        predictions = {
            "game_family": torch.tensor(
                [
                    [4.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0, 0.0],
                    [0.0, 4.0, 0.0, 0.0, 0.0],
                ]
            )
        }
        targets = {"game_family": torch.tensor([0, 3, 2])}
        masks = {"game_family": torch.tensor([True, True, False])}

        loss = compute_auxiliary_loss(
            predictions,
            targets,
            masks,
            weights={"game_family": 1.0},
        )

        self.assertLess(float(loss.terms["game_family"]), 0.1)
        self.assertEqual(2.0, float(loss.valid_counts["game_family"]))

    def test_missing_auxiliary_masks_make_zero_loss(self):
        predictions = {"death": torch.tensor([0.0, 1.0])}

        loss = compute_auxiliary_loss(
            predictions,
            targets={"death": torch.tensor([1.0, 0.0])},
            masks={"death": torch.tensor([False, False])},
            weights={"death": 1.0},
        )

        self.assertTrue(torch.allclose(torch.tensor(0.0), loss.total))
        self.assertEqual(0.0, float(loss.valid_counts["death"]))
