"""Mario action-set resolution for native NES and JoypadSpace policies."""
from collections.abc import Sequence
from dataclasses import dataclass

from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
    SIMPLE_MOVEMENT,
)


NATIVE_ACTION_SET = "nes"
NATIVE_ACTION_COUNT = 256
ACTION_SET_ALIASES = {"right": "right_only"}
ACTION_SETS = {
    NATIVE_ACTION_SET: None,
    "right": RIGHT_ONLY,
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


@dataclass(frozen=True)
class ResolvedActionSet:
    """Canonical action-space metadata for a configured Mario environment."""

    requested: str
    name: str
    actions: Sequence[Sequence[str]] | None
    num_actions: int
    native: bool = False


def resolve_action_set(
    action_set: str | Sequence[Sequence[str]],
    *,
    env=None,
) -> ResolvedActionSet:
    """Return canonical action metadata for a named or explicit action set."""
    if isinstance(action_set, str):
        requested = action_set
        key = action_set.strip().lower()
        canonical = ACTION_SET_ALIASES.get(key, key)
        try:
            actions = ACTION_SETS[key]
        except KeyError as exc:
            choices = ", ".join(sorted(ACTION_SETS))
            raise ValueError(f"unknown action set {action_set!r}; choose {choices}") from exc
        if canonical == NATIVE_ACTION_SET:
            return ResolvedActionSet(
                requested=requested,
                name=NATIVE_ACTION_SET,
                actions=None,
                num_actions=_env_action_count(env) or NATIVE_ACTION_COUNT,
                native=True,
            )
        return ResolvedActionSet(
            requested=requested,
            name=canonical,
            actions=actions,
            num_actions=len(actions),
            native=False,
        )

    actions = tuple(tuple(buttons) for buttons in action_set)
    return ResolvedActionSet(
        requested="custom",
        name="custom",
        actions=actions,
        num_actions=len(actions),
        native=False,
    )


def get_action_set(
    action_set: str | Sequence[Sequence[str]],
) -> Sequence[Sequence[str]] | None:
    """Return a JoypadSpace-compatible action list, or ``None`` for ``nes``."""
    return resolve_action_set(action_set).actions


def action_set_summary(
    action_set: str | Sequence[Sequence[str]],
    *,
    env=None,
) -> dict[str, int | str | bool]:
    """Return JSON-friendly action-space metadata for command payloads."""
    resolved = resolve_action_set(action_set, env=env)
    return {
        "action_set": resolved.name,
        "action_count": int(resolved.num_actions),
        "native_action_space": bool(resolved.native),
    }


def _env_action_count(env) -> int | None:
    if env is None:
        return None
    count = getattr(getattr(env, "action_space", None), "n", None)
    if count is None:
        return None
    return int(count)


__all__ = [
    "ACTION_SETS",
    "ACTION_SET_ALIASES",
    "NATIVE_ACTION_COUNT",
    "NATIVE_ACTION_SET",
    "ResolvedActionSet",
    "action_set_summary",
    "get_action_set",
    "resolve_action_set",
]
