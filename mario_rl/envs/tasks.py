"""Task metadata helpers for the gym-super-mario-bros 9.x environment surface."""
from __future__ import annotations

import random
from typing import Any

import gym_super_mario_bros


MarioTask = gym_super_mario_bros.MarioTask


def available_tasks(**filters: Any) -> tuple[MarioTask, ...]:
    """Return registered Mario task metadata matching gym-super-mario-bros filters."""
    return tuple(gym_super_mario_bros.all_tasks(**filters))


def available_env_ids(**filters: Any) -> tuple[str, ...]:
    """Return registered Mario environment IDs matching gym-super-mario-bros filters."""
    return tuple(gym_super_mario_bros.task_ids(**filters))


def task_for_env_id(env_id: str) -> MarioTask:
    """Return task metadata for a registered Mario environment ID."""
    return gym_super_mario_bros.task_for_env_id(env_id)


def choose_stage_env_id(
    *,
    seed: int | None = None,
    game_family: str = "smb1",
    include_aliases: bool = False,
    split: str = "train",
    validated: bool = True,
) -> str:
    """
    Select a deterministic single-stage environment ID from 9.x task metadata.

    gym-super-mario-bros 9.0.0 removed the old ``SuperMarioBrosRandomStages-*``
    registration family. Use this helper when a config needs seeded stage
    selection without relying on removed environment IDs.
    """
    candidates = available_env_ids(
        include_aliases=include_aliases,
        game_family=game_family,
        single_stage=True,
        split=split,
        validated=validated,
    )
    if not candidates:
        raise ValueError("no Mario stage environments match the requested filters")
    return random.Random(seed).choice(candidates)


__all__ = [
    "MarioTask",
    "available_env_ids",
    "available_tasks",
    "choose_stage_env_id",
    "task_for_env_id",
]
