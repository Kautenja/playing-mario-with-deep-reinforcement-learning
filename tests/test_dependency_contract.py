"""Runtime dependency contract tests for the modern Mario RL baseline."""
import importlib
from unittest import TestCase


class DependencyContract(TestCase):
    """Validate the active dependency set imports without legacy frameworks."""

    def test_modern_runtime_dependencies_import(self):
        for module_name in (
            "gymnasium",
            "gym_super_mario_bros",
            "nes_py",
            "torch",
            "lightning",
        ):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)
