"""Unit tests for the ReplyBuffer class."""
import os
import numpy as np
from unittest import TestCase
from ..replay_queue import ReplayQueue


# the name of the directory housing this module
DIR = os.path.dirname(os.path.realpath(__file__))


def arb_state():
    s = np.ones((10, 10, 4))
    a = 1
    r = 0
    d = False
    s2 = np.ones((10, 10, 4))
    return s, a, r, d, s2


def other_state():
    s = np.zeros((10, 10, 4))
    a = 3
    r = 1
    d = True
    s2 = np.zeros((10, 10, 4))
    return s, a, r, d, s2


def random_state(state=None):
    s = np.random.randint(0, 256, (10, 10, 4)).astype('uint8')
    a = np.random.randint(6)
    r = np.random.randint(2) - 1
    d = bool(np.random.randint(1))
    s2 = np.random.randint(0, 256, (10, 10, 4)).astype('uint8')
    return s, a, r, d, s2


class ReplyBuffer_init(TestCase):
    def test(self):
        arb = ReplayQueue()
        self.assertIsInstance(arb, object)


class ReplyBuffer_is_bound(TestCase):
    def test(self):
        arb = ReplayQueue(10)
        for i in range(25):
            # a different item is added so that it will be at the bottom
            # when the loop finishes
            if i == 15:
                arb.push(*other_state())
            else:
                arb.push(*arb_state())

        # there should only be 10 elements
        self.assertEqual(10, len(arb))
        # it should move the items along as new ones are added
        self.assertEqual(other_state()[1], arb.deque()[1])


class ReplyBuffer_len_(TestCase):
    def test(self):
        arb = ReplayQueue(10)
        self.assertEqual(0, len(arb))
        arb.push(*arb_state())
        self.assertEqual(1, len(arb))
        _ = arb.deque()
        self.assertEqual(0, len(arb))


class ReplyBuffer_sample(TestCase):
    def test(self):
        arb = ReplayQueue(1000)
        for i in range(1000):
            arb.push(*arb_state())

        batch = arb.sample()
        s, a, r, d, s2 = tuple(map(np.array, zip(*batch)))

        sample_size = len(s)

        exp_s, exp_a, exp_r, exp_d, exp_s2 = arb_state()

        self.assertEqual([exp_a] * sample_size, list(a))
        self.assertEqual([exp_r] * sample_size, list(r))
        self.assertEqual([exp_d] * sample_size, list(d))

        for index in range(sample_size):
            self.assertTrue(np.array_equal(exp_s, s[index]))
            self.assertTrue(np.array_equal(exp_s2, s2[index]))


class ReplyBuffer_sample_random(TestCase):
    def test(self):
        np.random.seed(1)
        arb = ReplayQueue(1000)
        for i in range(1000):
            arb.push(*random_state())

        batch = arb.sample()
        s, a, r, d, s2 = tuple(map(np.array, zip(*batch)))

        # save the arrays as the expected arrays
        # np.save('{}/arrays/s.npy'.format(DIR), s)
        # np.save('{}/arrays/a.npy'.format(DIR), a)
        # np.save('{}/arrays/r.npy'.format(DIR), r)
        # np.save('{}/arrays/d.npy'.format(DIR), d)
        # np.save('{}/arrays/s2.npy'.format(DIR), s2)

        # load the expected arrays
        exp_s = np.load('{}/arrays/s.npy'.format(DIR))
        exp_a = np.load('{}/arrays/a.npy'.format(DIR))
        exp_r = np.load('{}/arrays/r.npy'.format(DIR))
        exp_d = np.load('{}/arrays/d.npy'.format(DIR))
        exp_s2 = np.load('{}/arrays/s2.npy'.format(DIR))

        # check that the returned arrays are the expected ones
        self.assertTrue(np.array_equal(exp_s, s))
        self.assertTrue(np.array_equal(exp_a, a))
        self.assertTrue(np.array_equal(exp_r, r))
        self.assertTrue(np.array_equal(exp_d, d))
        self.assertTrue(np.array_equal(exp_s2, s2))
