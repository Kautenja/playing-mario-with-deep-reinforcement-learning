"""Unit tests for the Dueling Deep-Q model builder method."""
from unittest import TestCase
from keras.models import Model
from ..dueling_deep_q_model import build_dueling_deep_q_model


class ShouldBuildModel(TestCase):
    def test(self):
        model = build_dueling_deep_q_model()
        self.assertIsInstance(model, Model)
