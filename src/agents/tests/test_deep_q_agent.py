"""Unit tests for the DeepQAgent class."""
import gym
from unittest import TestCase
from ..deep_q_agent import DeepQAgent
from src.downsamplers import Downsampler


def arb_downsampler():
    return Downsampler((1, 1), (1, 1))


class DeepQAgent_init(TestCase):
    def test(self):
        arb = DeepQAgent(gym.make('SpaceInvaders-v0'), arb_downsampler())
        self.assertIsInstance(arb, object)


class DeepQAgent_is_subclass(TestCase):
    def test(self):
        arb = DeepQAgent(gym.make('SpaceInvaders-v0'), arb_downsampler())
        from ..agent import Agent
        self.assertIsInstance(arb, Agent)
