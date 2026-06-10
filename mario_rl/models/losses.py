"""PyTorch DQN and PPO loss helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical
from torch.nn import functional as F


@dataclass(frozen=True)
class PPOLoss:
    """Structured PPO loss terms for logging and tests."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


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


def compute_ppo_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float = 0.2,
    value_loss_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    normalize_advantages: bool = True,
) -> PPOLoss:
    """Return clipped PPO policy/value/entropy loss terms."""
    if policy_logits.ndim != 2:
        raise ValueError(
            "policy_logits must have shape (batch, actions), "
            f"got {tuple(policy_logits.shape)}"
        )
    device = policy_logits.device
    actions = actions.to(device=device, dtype=torch.long).view(-1)
    old_log_probabilities = old_log_probabilities.to(
        device=device,
        dtype=torch.float32,
    ).view(-1)
    returns = returns.to(device=device, dtype=torch.float32).view(-1)
    advantages = advantages.to(device=device, dtype=torch.float32).view(-1)
    values = values.to(device=device, dtype=torch.float32).view(-1)
    if actions.shape[0] != policy_logits.shape[0]:
        raise ValueError("actions batch dimension must match policy_logits")
    if normalize_advantages and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    distribution = Categorical(logits=policy_logits)
    log_probabilities = distribution.log_prob(actions)
    entropy = distribution.entropy().mean()
    log_ratio = log_probabilities - old_log_probabilities
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - float(clip_range),
        1.0 + float(clip_range),
    ) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(values, returns)
    total = (
        policy_loss
        + float(value_loss_coefficient) * value_loss
        - float(entropy_coefficient) * entropy
    )
    with torch.no_grad():
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (
            (torch.abs(ratio - 1.0) > float(clip_range))
            .to(dtype=torch.float32)
            .mean()
        )
    return PPOLoss(
        total=total,
        policy=policy_loss,
        value=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )
