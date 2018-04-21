import gym
from gym.spaces.multi_discrete import (
    BoxToMultiDiscrete,
    DiscreteToMultiDiscrete
)


class ToDiscreteWrapper(gym.Wrapper):
    """Wrapper to convert MultiDiscrete action space to Discrete."""

    def __init__(self, env):
        super(ToDiscreteWrapper, self).__init__(env)
        self.action_space = DiscreteToMultiDiscrete(self.action_space, {
            0:  [0, 0, 0, 0, 0, 0],  # NOOP
            1:  [1, 0, 0, 0, 0, 0],  # Up
            2:  [0, 0, 1, 0, 0, 0],  # Down
            3:  [0, 1, 0, 0, 0, 0],  # Left
            4:  [0, 1, 0, 0, 1, 0],  # Left + A
            5:  [0, 1, 0, 0, 0, 1],  # Left + B
            6:  [0, 1, 0, 0, 1, 1],  # Left + A + B
            7:  [0, 0, 0, 1, 0, 0],  # Right
            8:  [0, 0, 0, 1, 1, 0],  # Right + A
            9:  [0, 0, 0, 1, 0, 1],  # Right + B
            10: [0, 0, 0, 1, 1, 1],  # Right + A + B
            11: [0, 0, 0, 0, 1, 0],  # A
            12: [0, 0, 0, 0, 0, 1],  # B
            13: [0, 0, 0, 0, 1, 1],  # A + B
        })

    def _step(self, action):
        return self.env.step(self.action_space(action))


class ToBoxWrapper(gym.Wrapper):
    """Wrapper to convert MultiDiscrete action space to Box."""

    def __init__(self, env):
        super(ToBoxWrapper, self).__init__(env)
        self.action_space = BoxToMultiDiscrete(self.action_space)

    def _step(self, action):
        return self.env.step(self.action_space(action))


__all__ = [
    ToDiscreteWrapper.__name__,
    ToBoxWrapper.__name__
]
