"""Test cases for the annealing variable class."""
from unittest import TestCase
from ..annealing_variable import AnnealingVariable


class ShouldAnnealDecay(TestCase):
    def test(self):
        a = AnnealingVariable(1.0, 0.1, 1000)
        self.assertEqual(1.0, a.initial_value)
        self.assertEqual(1.0, a.value)
        self.assertEqual(0.1, a.final_value)
        self.assertEqual(0.9977000638225533, a.rate)

        for _ in range(500):
            a.step()
        self.assertAlmostEqual(0.3162277660168313, a.value)

        for _ in range(500):
            a.step()
        self.assertAlmostEqual(0.1, a.value)

        for _ in range(1000):
            a.step()
        self.assertAlmostEqual(0.1, a.value)


class ShouldAnnealGrow(TestCase):
    def test(self):
        a = AnnealingVariable(0.1, 1.0, 1000)
        self.assertEqual(0.1, a.initial_value)
        self.assertEqual(0.1, a.value)
        self.assertEqual(1.0, a.final_value)
        self.assertEqual(1 / 0.9977000638225533, a.rate)

        for _ in range(500):
            a.step()
        self.assertAlmostEqual(0.3162277660168292, a.value)

        for _ in range(500):
            a.step()
        self.assertAlmostEqual(1.0, a.value)

        for _ in range(1000):
            a.step()
        self.assertAlmostEqual(1.0, a.value)
