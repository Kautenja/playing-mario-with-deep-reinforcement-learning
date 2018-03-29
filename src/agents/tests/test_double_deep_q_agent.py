"""Unit tests for the DeepQAgent class."""
from unittest import TestCase
from ..double_deep_q_agent import DoubleDeepQAgent


class DoubleDeepQAgent_init(TestCase):
    def test(self):
        arb = DoubleDeepQAgent()
        self.assertIsInstance(arb, object)


class DoubleDeepQAgent_is_subclass(TestCase):
    def test(self):
        arb = DoubleDeepQAgent()
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
