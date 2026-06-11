"""Runtime dependency contract tests for the modern Mario RL baseline."""
import importlib
import importlib.metadata
from unittest import TestCase


def _version_tuple(distribution_name):
    """Return the numeric release tuple for an installed distribution."""
    version = importlib.metadata.version(distribution_name)
    return tuple(int(part) for part in version.split(".")[:3])


class DependencyContract(TestCase):
    """Validate the active dependency set imports without legacy frameworks."""

    def test_modern_runtime_dependencies_import(self):
        for module_name in (
            "gymnasium",
            "gym_super_mario_bros",
            "nes_py",
            "torch",
            "lightning",
            "rich",
        ):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_mario_runtime_uses_9x_environment_stack(self):
        self.assertGreaterEqual(_version_tuple("gym-super-mario-bros"), (9, 0, 0))
        self.assertGreaterEqual(_version_tuple("nes-py"), (9, 0, 0))
