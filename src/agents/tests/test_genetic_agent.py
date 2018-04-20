from unittest import TestCase
from ..genetic_agent import GeneticAgent
from src.environment.atari import build_atari_environment


class Should__init__(TestCase):
    def test(self):
        arb = GeneticAgent(build_atari_environment('Pong'))
        self.assertIsInstance(arb, GeneticAgent)


class Should__repr__(TestCase):
    def test(self):
        arb = GeneticAgent(build_atari_environment('Pong'))
        _R = "GeneticAgent(env=<FrameStackEnv<ClipRewardEnv<PenalizeDeathEnv<DownsampleEnv<FireResetEnv<MaxFrameskipEnv<NoopResetEnv<TimeLimit<AtariEnv<PongNoFrameskip-v4>>>>>>>>>>, render_mode='rgb_array')"
        self.assertEquals(_R, repr(arb))
