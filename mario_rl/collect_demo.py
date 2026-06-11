"""Collect pixel-only human demonstrations for imitation pretraining."""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mario_rl.config import (
    MarioRLConfig,
    action_space_summary,
    parse_cli_config,
    with_resolved_model_num_actions,
)
from mario_rl.envs.actions import resolve_action_set


@dataclass(frozen=True)
class KeyInput:
    """Keyboard state sampled for a single collection step."""

    pressed_keys: tuple[int, ...] = ()
    quit: bool = False
    reset: bool = False


KeyReader = Callable[[np.ndarray | None], KeyInput | Sequence[int] | int]


@dataclass(frozen=True)
class DemoCollectionOptions:
    """Runtime options for human demonstration collection."""

    output_dir: str | Path | None = None
    episodes: int = 1
    max_steps: int = 5000
    fps: float = 30.0
    repeat_last: bool = False
    source_notes: str = "human keyboard demonstration"
    window_name: str = "mario-rl demo"


def run(
    config: MarioRLConfig,
    *,
    options: DemoCollectionOptions | None = None,
    env_factory=None,
    key_reader: KeyReader | None = None,
) -> dict[str, Any]:
    """Collect human actions into one imitation ``.npz`` segment."""
    config = with_resolved_model_num_actions(config)
    options = options or DemoCollectionOptions()
    _validate_supported_collection_config(config)

    output_dir = Path(options.output_dir or config.imitation.data_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if env_factory is None:
        from dataclasses import replace

        from mario_rl.envs import make_env

        env_config = replace(
            config.env,
            render_mode="rgb_array",
            video_enabled=False,
        )

        def env_factory(active_config: MarioRLConfig):
            return make_env(config=env_config.to_mario_env_config())

    observations: list[np.ndarray] = []
    actions: list[int] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    boundaries: list[bool] = []

    env = env_factory(config)
    keys_to_action = _keys_to_action(env, config.env.action_set)
    reader = key_reader or PygletKeyboardReader(
        window_name=options.window_name,
        fps=options.fps,
        action_keys=_action_keys(keys_to_action),
    )
    _print_controls(keys_to_action, repeat_last=options.repeat_last)
    current_action = 0
    completed_episodes = 0
    total_steps = 0
    reset_seed = config.env.seed
    try:
        obs, _info = env.reset(seed=reset_seed)
        while completed_episodes < int(options.episodes) and total_steps < int(options.max_steps):
            frame = _render_frame(env)
            key_input = _coerce_key_input(reader(frame))
            if key_input.quit:
                break
            if key_input.reset:
                obs, _info = env.reset(seed=reset_seed)
                current_action = 0
                completed_episodes += 1
                continue

            if key_input.pressed_keys in keys_to_action:
                current_action = keys_to_action[key_input.pressed_keys]
            elif not options.repeat_last:
                current_action = 0

            observations.append(np.asarray(obs, dtype=np.uint8))
            actions.append(int(current_action))
            obs, _reward, terminated, truncated, _info = env.step(current_action)
            terminated = bool(terminated)
            truncated = bool(truncated)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            boundaries.append(terminated or truncated)
            total_steps += 1

            if terminated or truncated:
                completed_episodes += 1
                current_action = 0
                if completed_episodes < int(options.episodes):
                    obs, _info = env.reset(seed=reset_seed)
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()
        env.close()

    if not actions:
        raise RuntimeError("no demonstration steps were collected")

    action_summary = action_space_summary(config)
    observation_array = np.stack(observations).astype(np.uint8, copy=False)
    action_array = np.asarray(actions, dtype=np.int64)
    terminated_array = np.asarray(terminated_flags, dtype=np.bool_)
    truncated_array = np.asarray(truncated_flags, dtype=np.bool_)
    boundary_array = np.asarray(boundaries, dtype=np.bool_)
    metadata = _metadata(config, action_summary, observation_array, options)
    path = output_dir / _demo_filename(config)
    np.savez(
        path,
        observations=observation_array,
        actions=action_array,
        terminated=terminated_array,
        truncated=truncated_array,
        episode_boundaries=boundary_array,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    payload = {
        "command": "collect-demo",
        "output": str(path),
        "steps": int(action_array.shape[0]),
        "episodes": int(completed_episodes),
        **action_summary,
        "observation_shape": list(observation_array.shape[1:]),
        "metadata": metadata,
    }
    print(json.dumps(payload, sort_keys=True))
    return payload


class PygletKeyboardReader:
    """Display frames and return held keys using nes-py human-play semantics."""

    KEY_MAP: dict[int, int]

    def __init__(
        self,
        *,
        window_name: str,
        fps: float,
        action_keys: set[int],
    ) -> None:
        import pyglet

        self.pyglet = pyglet
        self.window_name = str(window_name)
        self.frame_duration = 1.0 / max(float(fps), 1.0)
        self.next_frame_time = time.monotonic()
        self.action_keys = set(action_keys)
        self.relevant_keys = set(action_keys) | {ord("t")}
        self.KEY_MAP = {
            self.pyglet.window.key.ENTER: ord("\r"),
            self.pyglet.window.key.SPACE: ord(" "),
        }
        self.window = None
        self.pressed_keys: set[int] = set()
        self.escape_pressed = False
        self.closed = False
        self.reset_latched = False

    def __call__(self, frame: np.ndarray | None) -> KeyInput:
        self._pace()
        if frame is not None:
            self._show(np.asarray(frame))
        elif self.window is not None:
            self.window.dispatch_events()

        reset_pressed = ord("t") in self.pressed_keys
        reset = reset_pressed and not self.reset_latched
        self.reset_latched = reset_pressed
        action_keys = tuple(sorted(key for key in self.pressed_keys if key in self.action_keys))
        return KeyInput(
            pressed_keys=action_keys,
            quit=bool(self.escape_pressed or self.closed),
            reset=reset,
        )

    def _pace(self) -> None:
        now = time.monotonic()
        if self.next_frame_time > now:
            time.sleep(self.next_frame_time - now)
            now = time.monotonic()
        self.next_frame_time = max(now, self.next_frame_time) + self.frame_duration

    def _show(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("demo render frame must have RGB shape (height, width, 3)")
        if self.window is None:
            self.window = self.pyglet.window.Window(
                caption=self.window_name,
                height=int(frame.shape[0]),
                width=int(frame.shape[1]),
                vsync=False,
                resizable=True,
            )
            self.window.event(self.on_key_press)
            self.window.event(self.on_key_release)
            self.window.event(self.on_close)

        self.pyglet.clock.tick()
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image = self.pyglet.image.ImageData(
            int(frame.shape[1]),
            int(frame.shape[0]),
            "RGB",
            frame.tobytes(),
            pitch=int(frame.shape[1]) * -3,
        )
        image.blit(0, 0, width=self.window.width, height=self.window.height)
        self.window.flip()

    def on_key_press(self, symbol, _modifiers) -> None:
        self._handle_key_event(symbol, True)

    def on_key_release(self, symbol, _modifiers) -> None:
        self._handle_key_event(symbol, False)

    def on_close(self) -> None:
        self.closed = True
        window = self.window
        self.window = None
        if window is not None:
            window.close()

    def _handle_key_event(self, symbol, is_press: bool) -> None:
        symbol = self.KEY_MAP.get(symbol, symbol)
        if symbol == self.pyglet.window.key.ESCAPE:
            self.escape_pressed = is_press
            return
        if symbol not in self.relevant_keys:
            return
        if is_press:
            self.pressed_keys.add(symbol)
        else:
            self.pressed_keys.discard(symbol)

    def close(self) -> None:
        window = self.window
        self.window = None
        if window is not None:
            window.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse collection options plus config overrides and collect a demo."""
    parser = argparse.ArgumentParser(
        prog="python -m mario_rl.collect_demo",
        description="Collect pixel-only human demonstrations for imitation pretraining.",
        epilog=(
            "Config overrides are passed through after collector options, for "
            "example: --config smb_ppo_imitation_fast_dev "
            "--train.fast_dev_run false."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--repeat-last", action="store_true")
    parser.add_argument("--no-repeat-last", action="store_false", dest="repeat_last")
    parser.add_argument("--source-notes", default="human keyboard demonstration")
    parser.add_argument("--window-name", default="mario-rl demo")
    parser.set_defaults(repeat_last=False)
    args, config_args = parser.parse_known_args(argv)

    config = parse_cli_config(config_args)
    options = DemoCollectionOptions(
        output_dir=args.output_dir,
        episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        repeat_last=bool(args.repeat_last),
        source_notes=args.source_notes,
        window_name=args.window_name,
    )
    run(config, options=options)
    return 0


def _validate_supported_collection_config(config: MarioRLConfig) -> None:
    if bool(config.env.macro_actions):
        raise ValueError("human demo collection currently expects env.macro_actions=false")
    summary = action_space_summary(config)
    if bool(summary["native_action_space"]):
        raise ValueError("human demo collection requires a Joypad action set, not native NES")
    shape = tuple(int(value) for value in config.replay.state_shape)
    if int(config.model.input_channels) != shape[0]:
        raise ValueError("model.input_channels must match replay.state_shape channels")


NES_PY_BUTTON_KEYS = {
    "right": ord("d"),
    "left": ord("a"),
    "down": ord("s"),
    "up": ord("w"),
    "start": ord("\r"),
    "select": ord(" "),
    "B": ord("p"),
    "A": ord("o"),
}


def _keys_to_action(env, action_set: str) -> dict[tuple[int, ...], int]:
    fallback = _keys_to_action_for_action_set(action_set)
    expected_actions = set(fallback.values())
    for candidate in _env_keymaps(env):
        if set(candidate.values()) == expected_actions:
            return candidate
    return fallback


def _env_keymaps(env) -> list[dict[tuple[int, ...], int]]:
    results = []
    visited: set[int] = set()
    current = env
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        for cls in type(current).__mro__:
            method = cls.__dict__.get("get_keys_to_action")
            if method is None:
                continue
            results.append(_normalize_keys_to_action(method(current)))
            break
        current = getattr(current, "env", None)
    return results


def _keys_to_action_for_action_set(action_set: str) -> dict[tuple[int, ...], int]:
    actions = resolve_action_set(action_set).actions
    if actions is None:
        raise ValueError("native NES action collection is not supported")
    result: dict[tuple[int, ...], int] = {}
    for index, buttons in enumerate(actions):
        if _button_key(buttons) == ("NOOP",):
            result[()] = int(index)
            continue
        result[tuple(sorted(_key_for_button(button) for button in buttons))] = int(index)
    return result


def _normalize_keys_to_action(mapping: Mapping[Sequence[int], int]) -> dict[tuple[int, ...], int]:
    return {
        tuple(sorted(int(key) for key in keys)): int(action)
        for keys, action in mapping.items()
    }


def _key_for_button(button: str) -> int:
    key = str(button)
    try:
        return NES_PY_BUTTON_KEYS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(NES_PY_BUTTON_KEYS))
        raise ValueError(f"unknown NES button {button!r}; choose one of: {choices}") from exc


def _action_keys(keys_to_action: Mapping[Sequence[int], int]) -> set[int]:
    return {int(key) for keys in keys_to_action for key in keys}


def _coerce_key_input(raw: KeyInput | Sequence[int] | int) -> KeyInput:
    if isinstance(raw, KeyInput):
        return raw
    if isinstance(raw, int):
        command = _command_for_key(raw)
        if command == "quit":
            return KeyInput(quit=True)
        if command == "reset":
            return KeyInput(reset=True)
        if raw < 0 or raw == 255:
            return KeyInput()
        return KeyInput(pressed_keys=(int(raw),))
    return KeyInput(pressed_keys=tuple(sorted(int(key) for key in raw)))


def _button_key(buttons: Sequence[str]) -> tuple[str, ...]:
    if len(buttons) == 1 and str(buttons[0]).upper() == "NOOP":
        return ("NOOP",)
    return tuple(sorted(str(button) for button in buttons))


def _command_for_key(key: int) -> str | None:
    if key == 27:
        return "quit"
    if key in (ord("t"), ord("T")):
        return "reset"
    return None


def _render_frame(env) -> np.ndarray | None:
    render = getattr(env, "render", None)
    if not callable(render):
        return None
    frame = render()
    if frame is None:
        return None
    return np.asarray(frame)


def _metadata(
    config: MarioRLConfig,
    action_summary: dict[str, Any],
    observations: np.ndarray,
    options: DemoCollectionOptions,
) -> dict[str, Any]:
    return {
        "env_id": config.env.id,
        "action_set": action_summary["action_set"],
        "action_count": int(action_summary["action_count"]),
        "macro_actions": bool(action_summary["macro_actions_enabled"]),
        "macro_action_set": action_summary.get("macro_action_set")
        or config.env.macro_action_set,
        "pixel_profile": config.env.pixel_profile,
        "observation_shape": [int(value) for value in observations.shape[1:]],
        "image_size": [int(value) for value in config.env.image_size],
        "frame_stack": int(config.env.frame_stack or 1),
        "channel_first": bool(config.env.channel_first),
        "source_notes": str(options.source_notes),
    }


def _demo_filename(config: MarioRLConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    env_slug = "".join(
        char.lower() if char.isalnum() else "-"
        for char in str(config.env.id)
    ).strip("-")
    return f"{timestamp}--{env_slug or 'mario'}--human-demo.npz"


def _print_controls(keys_to_action: Mapping[Sequence[int], int], *, repeat_last: bool) -> None:
    controls = [
        "Controls match nes-py human mode: d=right, a=left, s=down/crouch, w=up/door,",
        "          p=B/run, o=A/jump. Hold keys together for combos:",
        "          d+p=run right, d+o=right+jump, d+o+p=right+run+jump.",
        "          Enter=start and Space=select when exposed; t=manual reset; Esc=finish and save.",
        f"Repeat-last is {'on' if repeat_last else 'off'}; no held action defaults to NOOP.",
        f"Available bound key combos: {_format_bound_key_combos(keys_to_action)}",
    ]
    for line in controls:
        print(line)


def _format_bound_key_combos(keys_to_action: Mapping[Sequence[int], int]) -> list[str]:
    return [
        f"{_format_key_combo(keys)}={int(action)}"
        for keys, action in sorted(keys_to_action.items(), key=lambda item: int(item[1]))
    ]


def _format_key_combo(keys: Sequence[int]) -> str:
    labels = [_key_label(int(key)) for key in keys]
    return "+".join(labels) if labels else "NOOP"


def _key_label(key: int) -> str:
    if key == ord("\r"):
        return "Enter"
    if key == ord(" "):
        return "Space"
    if 32 <= key <= 126:
        return chr(key)
    return str(key)


if __name__ == "__main__":
    raise SystemExit(main())
