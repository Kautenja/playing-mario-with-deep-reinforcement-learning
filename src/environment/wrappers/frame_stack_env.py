"""An environment wrapper to stack observations into a tensor."""
from collections import deque
import numpy as np
import gym


class FrameStackEnv(gym.Wrapper):
    """An environment wrapper to stack observations into a tensor."""

    def __init__(self, env, history: int) -> None:
        """
        Initialize and environment that stacks previous frames.

        Args:
            env: the env to wrap around
            history: the length of the history

        Returns:
            None

        """
        gym.Wrapper.__init__(self, env)
        self.history = history
        # setup the new shape based on the history parameter
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * history),
            dtype=np.uint8
        )
        # setup the frame buffer using a dequeue
        self.frames = deque([], maxlen=history)

    def reset(self):
        frame = self.env.reset()
        for _ in range(self.history):
            self.frames.append(frame)
        return self._get_ob()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.frames.append(frame)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.history
        return np.concatenate(self.frames, axis=2)


# explicitly define the outward facing API of this module
__all__ = [FrameStackEnv.__name__]
