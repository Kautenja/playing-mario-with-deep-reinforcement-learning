"""Unit tests for the ReplyBuffer class."""
import numpy as np
from unittest import TestCase
from ..replay_queue import ReplayQueue


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


# class ReplyBuffer_sample(TestCase):
#     def test(self):
#         arb = ReplayQueue(1000)
#         for i in range(1000):
#             arb.push(*arb_state())

#         batch = arb.sample()
#         s, a, r, d, s2 = zip(*batch)

#         exp_s, exp_a, exp_r, exp_d, exp_s2 = arb_state()
#         # self.assertEqual([exp_s] * 32, list(s))
#         self.assertEqual([exp_a] * 32, list(a))
#         self.assertEqual([exp_r] * 32, list(r))
#         self.assertEqual([exp_d] * 32, list(d))
#         # self.assertEqual([exp_s2] * 32, list(s2))
