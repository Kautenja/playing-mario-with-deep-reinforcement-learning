"""Unit tests for the A3C_Agent class."""
from unittest import TestCase
from ..a3c_agent import A3C_Agent


class A3C_Agent_init(TestCase):
    def test(self):
        arb = A3C_Agent()
        self.assertIsInstance(arb, object)


class A3C_Agent_is_subclass(TestCase):
    def test(self):
        arb = A3C_Agent()
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
