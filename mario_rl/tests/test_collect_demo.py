"""Human demonstration collection contract tests."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from mario_rl.collect_demo import (
    DemoCollectionOptions,
    KeyInput,
    PygletKeyboardReader,
    _command_for_key,
    _collection_step_duration,
    _keys_to_action,
    _keys_to_action_for_action_set,
    run as run_collect_demo,
)
from mario_rl.imitation import load_imitation_dataset
from mario_rl.tests.fakes import FakeMarioEnv, tiny_ppo_config


class RenderableFakeMarioEnv(FakeMarioEnv):
    """Fake training env that exposes an RGB render surface."""

    def render(self):
        return np.full((240, 256, 3), self.step_count % 256, dtype=np.uint8)


class NativeKeymapFakeMarioEnv(RenderableFakeMarioEnv):
    """Fake Joypad-wrapped env that accidentally exposes raw NES key labels."""

    def get_keys_to_action(self):
        return {
            (): 0,
            (ord("d"),): 128,
            tuple(sorted((ord("d"), ord("o"), ord("p")))): 131,
        }


class CollectDemoTest(TestCase):
    """Validate demo collection writes loader-compatible imitation files."""

    def test_collect_demo_writes_pixel_only_npz_segment(self):
        with TemporaryDirectory() as tmpdir:
            config = tiny_ppo_config(tmpdir)
            data_dir = Path(tmpdir) / "demos"
            keys = iter([
                KeyInput(pressed_keys=(ord("d"),)),
                KeyInput(pressed_keys=tuple(sorted((ord("d"), ord("o"), ord("p"))))),
                KeyInput(quit=True),
            ])

            def fake_env_factory(_config):
                return RenderableFakeMarioEnv(
                    env_id=config.env.id,
                    num_actions=12,
                    episode_length=8,
                )

            def fake_key_reader(_frame):
                return next(keys)

            output = io.StringIO()
            with redirect_stdout(output):
                payload = run_collect_demo(
                    config,
                    options=DemoCollectionOptions(
                        output_dir=data_dir,
                        max_steps=8,
                    ),
                    env_factory=fake_env_factory,
                    key_reader=fake_key_reader,
                )

            path = Path(payload["output"])
            self.assertTrue(path.is_file())
            self.assertEqual(2, payload["steps"])
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(
                    {"observations", "actions", "terminated", "truncated", "episode_boundaries", "metadata"},
                    set(data.files),
                )
                self.assertEqual([1, 4], data["actions"].tolist())
                metadata = json.loads(str(data["metadata"].reshape(-1)[0].item()))
            self.assertEqual("complex", metadata["action_set"])
            self.assertEqual(12, metadata["action_count"])

            dataset = load_imitation_dataset(config, data_dir=data_dir)
            self.assertEqual(2, len(dataset))
            self.assertEqual(("FakeMario-v0",), dataset.env_ids)

    def test_complex_keys_match_nes_py_human_mode(self):
        keys_to_action = _keys_to_action_for_action_set("complex")

        self.assertEqual(0, keys_to_action[()])
        self.assertEqual(1, keys_to_action[(ord("d"),)])
        self.assertEqual(2, keys_to_action[tuple(sorted((ord("d"), ord("o"))))])
        self.assertEqual(3, keys_to_action[tuple(sorted((ord("d"), ord("p"))))])
        self.assertEqual(4, keys_to_action[tuple(sorted((ord("d"), ord("o"), ord("p"))))])
        self.assertEqual(5, keys_to_action[(ord("o"),)])
        self.assertNotIn((ord("p"),), keys_to_action)
        self.assertIsNone(_command_for_key(ord("p")))
        self.assertIsNone(_command_for_key(ord("P")))
        self.assertEqual("quit", _command_for_key(27))

    def test_native_keymap_is_rejected_for_complex_collection(self):
        env = NativeKeymapFakeMarioEnv(num_actions=12)
        keys_to_action = _keys_to_action(env, "complex")

        self.assertEqual(12, len(keys_to_action))
        self.assertEqual(1, keys_to_action[(ord("d"),)])
        self.assertEqual(4, keys_to_action[tuple(sorted((ord("d"), ord("o"), ord("p"))))])
        self.assertEqual(set(range(12)), set(keys_to_action.values()))

    def test_fps_caps_native_frames_not_collector_steps(self):
        config = tiny_ppo_config("/tmp")
        config = replace(config, env=replace(config.env, frame_skip=4))

        self.assertAlmostEqual(4.0 / 60.0, _collection_step_duration(config, 60.0))
        self.assertAlmostEqual(4.0 / 30.0, _collection_step_duration(config, 30.0))

    def test_escape_finish_is_latched_until_sampled(self):
        reader = PygletKeyboardReader(
            window_name="test",
            step_duration=0.0,
            action_keys=set(),
        )
        try:
            reader._handle_key_event(reader.pyglet.window.key.ESCAPE, True)
            reader._handle_key_event(reader.pyglet.window.key.ESCAPE, False)

            self.assertTrue(reader(None).quit)
        finally:
            reader.close()

    def test_window_close_during_dispatch_exits_render_cleanly(self):
        reader = PygletKeyboardReader(
            window_name="test",
            step_duration=0.0,
            action_keys=set(),
        )

        class ClosingWindow:
            width = 1
            height = 1

            def clear(self):
                pass

            def switch_to(self):
                pass

            def dispatch_events(self):
                reader.on_close()

            def flip(self):
                raise AssertionError("closed windows must not be flipped")

            def close(self):
                pass

        try:
            reader.window = ClosingWindow()
            reader._show(np.zeros((1, 1, 3), dtype=np.uint8))

            self.assertTrue(reader.closed)
            self.assertIsNone(reader.window)
            self.assertTrue(reader(None).quit)
        finally:
            reader.close()
