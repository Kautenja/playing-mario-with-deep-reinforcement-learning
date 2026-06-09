"""Package metadata and import-layout contract tests."""
import importlib
import sys
import tomllib
from pathlib import Path
from unittest import TestCase


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutContract(TestCase):
    """Validate the modern package surface without importing legacy modules."""

    def test_pyproject_declares_modern_package(self):
        pyproject = PROJECT_ROOT / "pyproject.toml"
        self.assertTrue(pyproject.is_file(), "pyproject.toml is required")

        with pyproject.open("rb") as pyproject_file:
            metadata = tomllib.load(pyproject_file)

        project = metadata["project"]
        self.assertEqual(">=3.13", project["requires-python"])
        self.assertIn("Programming Language :: Python :: 3.13", project["classifiers"])
        self.assertIn("Programming Language :: Python :: 3.14", project["classifiers"])

        package_finder = metadata["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(["mario_rl*"], package_finder["include"])

    def test_mario_rl_import_does_not_load_legacy_frameworks(self):
        sys.modules.pop("mario_rl", None)
        importlib.import_module("mario_rl")

        self.assertNotIn("tensorflow", sys.modules)
        self.assertNotIn("keras", sys.modules)
