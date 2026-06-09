"""Selected-action PyTorch DQN loss helpers."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def gather_action_q_values(q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return Q-values selected by integer action indices."""
    if q_values.ndim != 2:
        raise ValueError(f"q_values must have shape (batch, actions), got {tuple(q_values.shape)}")
    actions = actions.to(device=q_values.device, dtype=torch.long).view(-1, 1)
    if actions.shape[0] != q_values.shape[0]:
        raise ValueError("actions batch dimension must match q_values")
    return q_values.gather(dim=1, index=actions).squeeze(1)


def compute_td_targets(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    target_next_q_values: torch.Tensor,
    *,
    discount_factor: float,
    online_next_q_values: torch.Tensor | None = None,
    double_dqn: bool = False,
) -> torch.Tensor:
    """Compute one-step TD targets with explicit terminal and truncation masks."""
    rewards = rewards.to(device=target_next_q_values.device, dtype=torch.float32).view(-1)
    terminated = terminated.to(device=target_next_q_values.device, dtype=torch.bool).view(-1)
    truncated = truncated.to(device=target_next_q_values.device, dtype=torch.bool).view(-1)
    if double_dqn:
        if online_next_q_values is None:
            raise ValueError("online_next_q_values is required when double_dqn=True")
        next_actions = online_next_q_values.to(target_next_q_values.device).argmax(dim=1)
        next_q = gather_action_q_values(target_next_q_values, next_actions)
    else:
        next_q = target_next_q_values.max(dim=1).values
    done = terminated | truncated
    bootstrap = (~done).to(dtype=torch.float32)
    return rewards + float(discount_factor) * bootstrap * next_q


def compute_dqn_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return SmoothL1/Huber loss for selected action values."""
    selected_q = gather_action_q_values(q_values, actions)
    td_targets = td_targets.to(device=q_values.device, dtype=torch.float32).view(-1)
    return F.smooth_l1_loss(selected_q, td_targets, reduction=reduction)
