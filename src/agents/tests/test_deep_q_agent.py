"""Unit tests for the DeepQAgent class."""
from unittest import TestCase
from ..deep_q_agent import DeepQAgent


class DeepQAgent_init(TestCase):
    def test(self):
        arb = DeepQAgent()
        self.assertIsInstance(arb, object)


class DeepQAgent_is_subclass(TestCase):
    def test(self):
        arb = DeepQAgent()
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
