"""Unit tests for the ReplyBuffer class."""
import os
import numpy as np
from unittest import TestCase
from ..prioritized_replay_queue import PrioritizedReplayQueue


# the name of the directory housing this module
DIR = os.path.dirname(os.path.realpath(__file__))


def ones() -> tuple:
    """Return an arbitrary state of ones."""
    s = np.ones((84, 84, 4), dtype=np.uint8)
    a = 1
    r = 1
    d = True
    s2 = np.ones((84, 84, 4), dtype=np.uint8)
    return s, a, r, d, s2


def zeros() -> tuple:
    """Return an arbitrary state of zeros."""
    s = np.zeros((84, 84, 4), dtype=np.uint8)
    a = 0
    r = 0
    d = False
    s2 = np.zeros((84, 84, 4), dtype=np.uint8)
    return s, a, r, d, s2


def random_state() -> tuple:
    """Return an arbitrary randomized state"""
    s = np.random.randint(0, 256, (84, 84, 4)).astype(np.uint8)
    a = np.random.randint(6)
    r = np.random.randint(2) - 1
    d = bool(np.random.randint(1))
    s2 = np.random.randint(0, 256, (84, 84, 4)).astype(np.uint8)
    return s, a, r, d, s2


class ReplyBuffer__init__(TestCase):
    def test(self):
        self.assertIsInstance(PrioritizedReplayQueue(10), object)
        self.assertIsInstance(PrioritizedReplayQueue(size=10), object)


class ReplyBuffer__repr__(TestCase):
    def test(self):
        self.assertEqual('PrioritizedReplayQueue(size=4321)', repr(PrioritizedReplayQueue(4321)))
        self.assertEqual('PrioritizedReplayQueue(size=1234)', repr(PrioritizedReplayQueue(size=1234)))


class ReplyBuffer__len__(TestCase):
    def test(self):
        arb = PrioritizedReplayQueue(10)
        self.assertEqual(0, arb.top)
        arb.push(*zeros(), priority=0)
        self.assertEqual(1, arb.top)

        for index in range(2, 30):
            if index < 10:
                arb.push(*zeros(), priority=index)
                self.assertEqual(index, arb.top)
            else:
                arb.push(*ones(), priority=index)
                self.assertEqual(10, arb.top)


class ReplyBuffer_is_bound(TestCase):
    def test(self):
        arb = PrioritizedReplayQueue(10)
        for i in range(25):
            # a different item is added so that it will be at the bottom
            # when the loop finishes
            if i == 15:
                arb.push(*ones(), priority=i)
            else:
                arb.push(*zeros(), priority=i)

        # there should only be 10 elements
        self.assertEqual(10, arb.top)


class ReplyBuffer_sample(TestCase):
    def test(self):
        arb = PrioritizedReplayQueue(1000)
        for i in range(1000):
            arb.push(*ones(), priority=i)

        s, a, r, d, s2 = arb.sample()

        self.assertEqual(s.dtype, np.uint8)
        self.assertEqual(a.dtype, np.uint8)
        self.assertEqual(r.dtype, np.int8)
        self.assertEqual(d.dtype, np.bool)
        self.assertEqual(s2.dtype, np.uint8)

        sample_size = len(s)

        exp_s, exp_a, exp_r, exp_d, exp_s2 = ones()

        self.assertEqual([exp_a] * sample_size, list(a))
        self.assertEqual([exp_r] * sample_size, list(r))
        self.assertEqual([exp_d] * sample_size, list(d))

        for index in range(sample_size):
            self.assertTrue(np.array_equal(exp_s, s[index]))
            self.assertTrue(np.array_equal(exp_s2, s2[index]))


class ReplyBuffer_sample_random(TestCase):
    def test(self):
        np.random.seed(1)
        arb = PrioritizedReplayQueue(1000)
        for i in range(1000):
            arb.push(*random_state(), priority=i)

        s, a, r, d, s2 = arb.sample()

        self.assertEqual(s.dtype, np.uint8)
        self.assertEqual(a.dtype, np.uint8)
        self.assertEqual(r.dtype, np.int8)
        self.assertEqual(d.dtype, np.bool)
        self.assertEqual(s2.dtype, np.uint8)

        # save the arrays as the expected arrays
        # np.save('{}/arrays/s_np_prio.npy'.format(DIR), s)
        # np.save('{}/arrays/a_np_prio.npy'.format(DIR), a)
        # np.save('{}/arrays/r_np_prio.npy'.format(DIR), r)
        # np.save('{}/arrays/d_np_prio.npy'.format(DIR), d)
        # np.save('{}/arrays/s2_np_prio.npy'.format(DIR), s2)

        # load the expected arrays
        exp_s = np.load('{}/arrays/s_np_prio.npy'.format(DIR))
        exp_a = np.load('{}/arrays/a_np_prio.npy'.format(DIR))
        exp_r = np.load('{}/arrays/r_np_prio.npy'.format(DIR))
        exp_d = np.load('{}/arrays/d_np_prio.npy'.format(DIR))
        exp_s2 = np.load('{}/arrays/s2_np_prio.npy'.format(DIR))

        # check that the returned arrays are the expected ones
        self.assertTrue(np.array_equal(exp_s, s))
        self.assertTrue(np.array_equal(exp_a, a))
        self.assertTrue(np.array_equal(exp_r, r))
        self.assertTrue(np.array_equal(exp_d, d))
        self.assertTrue(np.array_equal(exp_s2, s2))


class ReplyBuffer_should_push_items_with_uniform_priority(TestCase):
    def test(self):
        arb = PrioritizedReplayQueue(10)
        self.assertEqual(0, arb.top)
        arb.push(*zeros(), priority=0)
        self.assertEqual(1, arb.top)

        for index in range(2, 30):
            if index < 10:
                arb.push(*zeros(), priority=0)
                self.assertEqual(index, arb.top)
            else:
                arb.push(*ones(), priority=0)
                self.assertEqual(10, arb.top)
