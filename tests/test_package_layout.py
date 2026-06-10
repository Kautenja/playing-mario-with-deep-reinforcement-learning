"""Package metadata and import-layout contract tests."""
import importlib
import importlib.util
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
        self.assertEqual(["src", "src.*"], package_finder["exclude"])

    def test_mario_rl_import_does_not_load_legacy_frameworks(self):
        sys.modules.pop("mario_rl", None)
        importlib.import_module("mario_rl")

        self.assertNotIn("tensorflow", sys.modules)
        self.assertNotIn("keras", sys.modules)

    def test_repository_main_delegates_to_modern_package(self):
        for module_name in list(sys.modules):
            if module_name == "src" or module_name.startswith("src."):
                sys.modules.pop(module_name, None)

        main_path = PROJECT_ROOT / "__main__.py"
        spec = importlib.util.spec_from_file_location("repo_main_under_test", main_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIs(module.main, importlib.import_module("mario_rl.__main__").main)
        self.assertNotIn("src", sys.modules)
        self.assertNotIn("tensorflow", sys.modules)
        self.assertNotIn("keras", sys.modules)

    def test_legacy_src_package_is_explicitly_deprecated(self):
        for module_name in list(sys.modules):
            if module_name == "src" or module_name.startswith("src."):
                sys.modules.pop(module_name, None)

        with self.assertWarnsRegex(DeprecationWarning, "legacy src package is deprecated"):
            legacy = importlib.import_module("src")

        self.assertIn("mario_rl", legacy.DEPRECATION_MESSAGE)
        self.assertIn("deprecated", (PROJECT_ROOT / "src" / "README.md").read_text().lower())
        self.assertNotIn("tensorflow", sys.modules)
        self.assertNotIn("keras", sys.modules)
