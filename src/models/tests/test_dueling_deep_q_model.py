"""Unit tests for the Dueling Deep-Q model builder method."""
import importlib.util
import unittest
from unittest import TestCase

if importlib.util.find_spec("keras") is None:
    raise unittest.SkipTest("legacy Keras model tests require Keras")

from keras.models import Model
from ..dueling_deep_q_model import build_dueling_deep_q_model


class ShouldBuildModel(TestCase):
    def test(self):
        model = build_dueling_deep_q_model()
        self.assertIsInstance(model, Model)
