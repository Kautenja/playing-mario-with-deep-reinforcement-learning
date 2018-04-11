"""Unit tests for the A3C model builder method."""
import numpy as np
from unittest import TestCase
from keras.models import Model
from ..a3c_model import build_a3c_model


class ShouldBuildModel(TestCase):
    def test(self):
        model = build_a3c_model()
        self.assertIsInstance(model, Model)


class ShouldPassThroughZeros(TestCase):
    def test(self):
        model = build_a3c_model()
        s = np.zeros((1, 84, 84, 4))
        s_batch = np.zeros((32, 84, 84, 4))
        a_batch = np.zeros((32, 6))
        r_batch = np.zeros((32, ))
        p, v = model.predict([s, s_batch, a_batch, r_batch])

        self.assertEqual(1, p.sum())
        self.assertEqual(0, v.sum())


class ShouldPassThroughOnes(TestCase):
    def test(self):
        model = build_a3c_model()
        s = np.ones((1, 84, 84, 4))
        s_batch = np.ones((32, 84, 84, 4))
        a_batch = np.ones((32, 6))
        r_batch = np.ones((32, ))
        p, v = model.predict([s, s_batch, a_batch, r_batch])

        self.assertAlmostEqual(1, p.sum(), places=4)


class ShouldPassThroughRandom(TestCase):
    def test(self):
        model = build_a3c_model()
        s = np.random.random((1, 84, 84, 4))
        s_batch = np.random.random((32, 84, 84, 4))
        a_batch = np.random.random((32, 6))
        r_batch = np.random.random((32, ))
        p, v = model.predict([s, s_batch, a_batch, r_batch])

        self.assertAlmostEqual(1, p.sum(), places=4)

