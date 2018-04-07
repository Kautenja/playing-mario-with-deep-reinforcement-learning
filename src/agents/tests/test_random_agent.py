"""Unit tests for the RandomAgent class."""
import gym
from unittest import TestCase
from ..random_agent import RandomAgent


class RandomAgent_init(TestCase):
    def test(self):
        arb = RandomAgent(gym.make('SpaceInvaders-v0'))
        self.assertIsInstance(arb, object)


class RandomAgent_is_subclass(TestCase):
    def test(self):
        arb = RandomAgent(gym.make('SpaceInvaders-v0'))
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
