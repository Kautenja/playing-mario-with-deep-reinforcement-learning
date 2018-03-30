"""Unit tests for the ReplyBuffer class."""
from unittest import TestCase
from ..replay_queue import ReplayQueue


class ReplyBuffer_init(TestCase):
    def test(self):
        arb = ReplayQueue()
        self.assertIsInstance(arb, object)


class ReplyBuffer_is_bound(TestCase):
    def test(self):
        arb = ReplayQueue(10)
        for i in range(25):
            arb.push(i)

        # there should only be 10 elements
        self.assertEqual(10, len(arb))
        # it should move the items along as new ones are afdded
        self.assertEqual(15, *arb.dequeu())


class ReplyBuffer_len_(TestCase):
    def test(self):
        arb = ReplayQueue(10)
        self.assertEqual(0, len(arb))
        arb.push('arb')
        self.assertEqual(1, len(arb))
        _ = arb.dequeu()
        self.assertEqual(0, len(arb))


class ReplyBuffer_sample(TestCase):
    def test(self):
        arb = ReplayQueue(1000)
        for i in range(10):
            arb.push('s', 'a', 'r', 'd', 's2')

        s, a, r, d, s2 = arb.sample()

        self.assertEqual(['s'] * 10, list(s))
        self.assertEqual(['a'] * 10, list(a))
        self.assertEqual(['r'] * 10, list(r))
        self.assertEqual(['d'] * 10, list(d))
        self.assertEqual(['s2'] * 10, list(s2))
