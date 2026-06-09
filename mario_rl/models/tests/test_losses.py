"""PyTorch DQN loss and target contract tests."""
from __future__ import annotations

from unittest import TestCase

import torch

from mario_rl.models import DQN, compute_dqn_loss, compute_td_targets, gather_action_q_values


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
