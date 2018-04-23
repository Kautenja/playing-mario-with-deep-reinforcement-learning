"""A wrapper to map the multi-discrete actions of SMB to discrete actions."""
import gym


class ToDiscreteActionSpaceEnv(gym.Wrapper):
    """An adapter from multi-discrete Super Mario inputs to discrete ones."""

    def __init__(self, env) -> None:
        """
        Create a wrapper to map the action input from discrete inputs to
        the expected multi-discrete inputs.
        """
        super().__init__(env)
        self.discrete_to_multi = {
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
        }
        # assign a new action space to this wrapper for sampling from
        self.action_space = gym.spaces.Discrete(len(self.discrete_to_multi))

    def step(self, action):
        # unwrap the action using the map
        return self.env.step(self.discrete_to_multi[action])

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


__all__ = [ToDiscreteActionSpaceEnv.__name__]
