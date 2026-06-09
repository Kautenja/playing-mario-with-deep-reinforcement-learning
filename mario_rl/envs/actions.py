"""Mario action-set resolution for JoypadSpace."""
from collections.abc import Sequence

from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
    SIMPLE_MOVEMENT,
)


ACTION_SETS = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


def get_action_set(action_set: str | Sequence[Sequence[str]]) -> Sequence[Sequence[str]]:
    """Return a JoypadSpace-compatible action list."""
    if isinstance(action_set, str):
        try:
            return ACTION_SETS[action_set]
        except KeyError as exc:
            choices = ", ".join(sorted(ACTION_SETS))
            raise ValueError(f"unknown action set {action_set!r}; choose {choices}") from exc

    return action_set


__all__ = ["ACTION_SETS", "get_action_set"]
