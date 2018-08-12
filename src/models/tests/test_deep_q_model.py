"""Unit tests for the Deep-Q model builder method."""
from unittest import TestCase
from keras.models import Model
from ..deep_q_model import build_deep_q_model


class ShouldBuildModel(TestCase):
    def test(self):
        model = build_deep_q_model()
        self.assertIsInstance(model, Model)
