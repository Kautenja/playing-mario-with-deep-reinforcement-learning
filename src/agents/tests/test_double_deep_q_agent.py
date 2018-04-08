"""Unit tests for the DeepQAgent class."""
import gym
from unittest import TestCase
from ..double_deep_q_agent import DoubleDeepQAgent
from src.downsamplers import Downsampler


def arb_downsampler():
    return Downsampler((1, 1), (1, 1))


class DoubleDeepQAgent_init(TestCase):
    def test(self):
        arb = DoubleDeepQAgent(gym.make('SpaceInvaders-v0'), arb_downsampler())
        self.assertIsInstance(arb, object)


class DoubleDeepQAgent_is_subclass(TestCase):
    def test(self):
        arb = DoubleDeepQAgent(gym.make('SpaceInvaders-v0'), arb_downsampler())
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
