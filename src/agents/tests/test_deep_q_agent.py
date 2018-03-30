"""Unit tests for the DeepQAgent class."""
from unittest import TestCase
from src.models import build_deep_mind_model
from ..deep_q_agent import DeepQAgent


class DeepQAgent_init(TestCase):
    def test(self):
        arb = DeepQAgent(build_deep_mind_model())
        self.assertIsInstance(arb, object)


class DeepQAgent_is_subclass(TestCase):
    def test(self):
        arb = DeepQAgent(build_deep_mind_model())
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
