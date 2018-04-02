"""Unit tests for the DeepQAgent class."""
import gym
import numpy as np
from unittest import TestCase
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


class DeepQAgent_save_load(TestCase):
    def test(self):
        saved = DeepQAgent(gym.make('SpaceInvaders-v0'))
        loaded = DeepQAgent(gym.make('SpaceInvaders-v0'))
        # compare weights and ensure they are different
        w1 = saved.model.get_weights()
        w2 = loaded.model.get_weights()
        for _w1, _w2 in zip(w1, w2):
            # ignore blank bias matrices
            if _w1.sum() == 0 and _w2.sum() == 0:
                continue
            self.assertFalse(np.array_equal(_w1, _w2))

        # load the weights and ensure they are the same
        saved.save_model()
        loaded.load_model()
        w1 = saved.model.get_weights()
        w2 = loaded.model.get_weights()
        for _w1, _w2 in zip(w1, w2):
            self.assertTrue(np.array_equal(_w1, _w2))




