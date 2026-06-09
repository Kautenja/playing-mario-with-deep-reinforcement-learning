"""Config package test exports for ``python -m unittest mario_rl.config.tests``."""
from .test_config import ConfigCliTest, ConfigSchemaTest


__all__ = ["ConfigCliTest", "ConfigSchemaTest"]
