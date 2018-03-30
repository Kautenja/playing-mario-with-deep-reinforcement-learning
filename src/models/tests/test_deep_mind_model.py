"""Unit tests for the DeepMind model builder method."""
from unittest import TestCase
from keras.models import Model
from ..deep_mind_model import build_deep_mind_model


class ShouldBuildModel(TestCase):
    def test(self):
        model = build_deep_mind_model()
        self.assertIsInstance(model, Model)

