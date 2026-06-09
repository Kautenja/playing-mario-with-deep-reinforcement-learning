"""Unit tests for the Deep-Q model builder method."""
import importlib.util
import unittest
from unittest import TestCase

if importlib.util.find_spec("keras") is None:
    raise unittest.SkipTest("legacy Keras model tests require Keras")

from keras.models import Model
from ..deep_q_model import build_deep_q_model


class ShouldBuildModel(TestCase):
    def test(self):
        model = build_deep_q_model()
        self.assertIsInstance(model, Model)
