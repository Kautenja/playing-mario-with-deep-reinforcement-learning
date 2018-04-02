"""Unit tests for the DeepQAgent class."""
from unittest import TestCase
import gym
from ..deep_q_agent import DeepQAgent


class DeepQAgent_init(TestCase):
    def test(self):
        arb = DeepQAgent(gym.make('SpaceInvaders-v0'))
        self.assertIsInstance(arb, object)


class DeepQAgent_is_subclass(TestCase):
    def test(self):
        arb = DeepQAgent(gym.make('SpaceInvaders-v0'))
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
