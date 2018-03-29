"""Unit tests for the Agent abstract base class."""
from unittest import TestCase
from ..agent import Agent


class Agent_init(TestCase):
    def test(self):
        arb = Agent()
        self.assertIsInstance(arb, object)
        self.assertIsInstance(arb, Agent)
