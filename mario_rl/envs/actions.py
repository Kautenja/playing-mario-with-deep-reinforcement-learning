"""Mario action-set and macro-action resolution."""
from collections.abc import Sequence
from dataclasses import dataclass

from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
    SIMPLE_MOVEMENT,
)


NATIVE_ACTION_SET = "nes"
NATIVE_ACTION_COUNT = 256
DEFAULT_MACRO_ACTION_SET = "conservative"
ACTION_SET_ALIASES = {"right": "right_only"}
ACTION_SETS = {
    NATIVE_ACTION_SET: None,
    "right": RIGHT_ONLY,
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}
MACRO_ACTION_SET_ALIASES = {
    "default": DEFAULT_MACRO_ACTION_SET,
    "movement": DEFAULT_MACRO_ACTION_SET,
}
MACRO_ACTION_SET_DESCRIPTIONS = {
    DEFAULT_MACRO_ACTION_SET: (
        "Primitive Joypad actions plus short deterministic movement sequences "
        "for rightward running, jumping, braking, crouching, and waiting."
    ),
}


@dataclass(frozen=True)
class ResolvedActionSet:
    """Canonical action-space metadata for a configured Mario environment."""

    requested: str
    name: str
    actions: Sequence[Sequence[str]] | None
    num_actions: int
    native: bool = False


@dataclass(frozen=True)
class MacroAction:
    """A named deterministic sequence of resolved Joypad action indices."""

    name: str
    action_indices: tuple[int, ...]
    button_sequence: tuple[tuple[str, ...], ...]
    description: str = ""

    @property
    def length(self) -> int:
        """Return the number of Joypad actions in the sequence."""
        return len(self.action_indices)


@dataclass(frozen=True)
class ResolvedMacroActionSet:
    """Canonical metadata for a configured macro-action set."""

    requested: str
    name: str
    actions: tuple[MacroAction, ...]
    unavailable_actions: tuple[str, ...] = ()

    @property
    def num_actions(self) -> int:
        """Return the number of macro actions exposed to the policy."""
        return len(self.actions)


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


def resolve_macro_action_set(
    macro_action_set: str,
    base_action_set: ResolvedActionSet,
) -> ResolvedMacroActionSet:
    """Resolve a named macro-action set against concrete Joypad actions."""
    if base_action_set.native or base_action_set.actions is None:
        raise ValueError(
            "macro actions require a Joypad action set; use right_only, simple, "
            "or complex instead of native NES actions"
        )
    requested = str(macro_action_set)
    key = requested.strip().lower()
    canonical = MACRO_ACTION_SET_ALIASES.get(key, key)
    if canonical != DEFAULT_MACRO_ACTION_SET:
        choices = ", ".join(sorted(MACRO_ACTION_SET_DESCRIPTIONS))
        raise ValueError(f"unknown macro action set {macro_action_set!r}; choose {choices}")

    base_actions = tuple(tuple(buttons) for buttons in base_action_set.actions)
    button_index = {_button_key(buttons): index for index, buttons in enumerate(base_actions)}
    macros: list[MacroAction] = [
        MacroAction(
            name=f"primitive_{index}_{_buttons_slug(buttons)}",
            action_indices=(index,),
            button_sequence=(buttons,),
            description=f"Single Joypad action {index}: {_buttons_label(buttons)}.",
        )
        for index, buttons in enumerate(base_actions)
    ]
    unavailable: list[str] = []
    for name, alternatives, description in _conservative_macro_specs():
        macro = _resolve_macro_spec(name, alternatives, description, button_index, base_actions)
        if macro is None:
            unavailable.append(name)
            continue
        macros.append(macro)

    return ResolvedMacroActionSet(
        requested=requested,
        name=canonical,
        actions=tuple(macros),
        unavailable_actions=tuple(unavailable),
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
    macro_actions: bool = False,
    macro_action_set: str = DEFAULT_MACRO_ACTION_SET,
    frame_skip: int | None = None,
) -> dict[str, object]:
    """Return JSON-friendly action-space metadata for command payloads."""
    resolved = resolve_action_set(action_set, env=env)
    summary: dict[str, object] = {
        "action_set": resolved.name,
        "base_action_set": resolved.name,
        "base_action_count": int(resolved.num_actions),
        "action_count": int(resolved.num_actions),
        "native_action_space": bool(resolved.native),
        "macro_actions_enabled": False,
        "macro_action_set": None,
        "macro_action_count": 0,
        "macro_action_sequence_count": 0,
        "macro_actions": [],
        "macro_unavailable_actions": [],
    }
    if macro_actions:
        macro = resolve_macro_action_set(macro_action_set, resolved)
        summary.update(
            {
                "action_count": int(macro.num_actions),
                "macro_actions_enabled": True,
                "macro_action_set": macro.name,
                "macro_action_count": int(macro.num_actions),
                "macro_action_sequence_count": int(macro.num_actions),
                "macro_actions": [macro_action_summary(action) for action in macro.actions],
                "macro_unavailable_actions": list(macro.unavailable_actions),
            }
        )
    if frame_skip is not None:
        resolved_skip = max(int(frame_skip), 1)
        summary["frame_skip"] = resolved_skip
        summary["macro_frame_skip"] = resolved_skip
        summary["macro_frame_skip_interaction"] = (
            "Each macro step runs its Joypad index sequence; each Joypad index "
            f"then advances through frame_skip={resolved_skip} when frame skip is enabled. "
            "Reported frames_skipped is the executed aggregate, shortened by early "
            "termination or truncation."
        )
    return summary


def macro_action_summary(action: MacroAction) -> dict[str, object]:
    """Return JSON-friendly macro-action metadata."""
    return {
        "name": action.name,
        "length": int(action.length),
        "sequence": [int(index) for index in action.action_indices],
        "buttons": [list(buttons) for buttons in action.button_sequence],
        "description": action.description,
    }


def _env_action_count(env) -> int | None:
    if env is None:
        return None
    count = getattr(getattr(env, "action_space", None), "n", None)
    if count is None:
        return None
    return int(count)


def _conservative_macro_specs():
    right_b = ("right", "B")
    right_a = ("right", "A")
    right_ab = ("right", "A", "B")
    return (
        (
            "wait",
            (_repeat(("NOOP",), 4),),
            "Hold no buttons for four Joypad steps.",
        ),
        (
            "run_right",
            (_repeat(right_b, 8),),
            "Hold right and B for eight Joypad steps.",
        ),
        (
            "short_jump",
            (
                _repeat(("A",), 3) + _repeat(("NOOP",), 2),
                _repeat(right_a, 3) + _repeat(("right",), 2),
            ),
            "Tap jump briefly, falling back to right+jump on right-only sets.",
        ),
        (
            "full_jump",
            (
                _repeat(("A",), 8),
                _repeat(right_a, 8),
            ),
            "Hold jump for eight Joypad steps.",
        ),
        (
            "run_jump",
            (
                _repeat(right_b, 2) + _repeat(right_ab, 8),
            ),
            "Build rightward speed, then hold right, A, and B.",
        ),
        (
            "hold_left",
            (_repeat(("left",), 8),),
            "Hold left for eight Joypad steps.",
        ),
        (
            "crouch",
            (_repeat(("down",), 8),),
            "Hold down for eight Joypad steps when the Joypad set exposes it.",
        ),
    )


def _resolve_macro_spec(
    name: str,
    alternatives: Sequence[Sequence[Sequence[str]]],
    description: str,
    button_index: dict[tuple[str, ...], int],
    base_actions: Sequence[Sequence[str]],
) -> MacroAction | None:
    for alternative in alternatives:
        indices: list[int] = []
        buttons: list[tuple[str, ...]] = []
        for button_combo in alternative:
            index = button_index.get(_button_key(button_combo))
            if index is None:
                break
            indices.append(index)
            buttons.append(tuple(base_actions[index]))
        else:
            return MacroAction(
                name=name,
                action_indices=tuple(indices),
                button_sequence=tuple(buttons),
                description=description,
            )
    return None


def _repeat(buttons: Sequence[str], count: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(buttons) for _ in range(int(count)))


def _button_key(buttons: Sequence[str]) -> tuple[str, ...]:
    if len(buttons) == 1 and str(buttons[0]).upper() == "NOOP":
        return ("NOOP",)
    return tuple(sorted(str(button) for button in buttons))


def _buttons_slug(buttons: Sequence[str]) -> str:
    if len(buttons) == 1 and str(buttons[0]).upper() == "NOOP":
        return "noop"
    return "_".join(str(button).lower() for button in buttons)


def _buttons_label(buttons: Sequence[str]) -> str:
    if len(buttons) == 1 and str(buttons[0]).upper() == "NOOP":
        return "NOOP"
    return "+".join(str(button) for button in buttons)


__all__ = [
    "ACTION_SETS",
    "ACTION_SET_ALIASES",
    "DEFAULT_MACRO_ACTION_SET",
    "MACRO_ACTION_SET_ALIASES",
    "MACRO_ACTION_SET_DESCRIPTIONS",
    "NATIVE_ACTION_COUNT",
    "NATIVE_ACTION_SET",
    "MacroAction",
    "ResolvedActionSet",
    "ResolvedMacroActionSet",
    "action_set_summary",
    "get_action_set",
    "macro_action_summary",
    "resolve_action_set",
    "resolve_macro_action_set",
]
