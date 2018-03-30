"""Unit tests for the RandomAgent class."""
from unittest import TestCase
from ..random_agent import RandomAgent


class RandomAgent_init(TestCase):
    def test(self):
        arb = RandomAgent()
        self.assertIsInstance(arb, object)


class RandomAgent_is_subclass(TestCase):
    def test(self):
        arb = RandomAgent()
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
