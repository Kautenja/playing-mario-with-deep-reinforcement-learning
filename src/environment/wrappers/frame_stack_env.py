"""An environment wrapper to stack observations into a tensor."""
from collections import deque
import numpy as np
import gym


class FrameStackEnv(gym.Wrapper):
    """An environment wrapper to stack observations into a tensor."""

    def __init__(self, env, history_length: int=4) -> None:
        """
        Initialize a new frame stacking environment.

        Args:
            env: the environment to stack frames of
            history_length: the number of frames to stack

        """
        gym.Wrapper.__init__(self, env)
        self.history_length = history_length
        # unwrap the new shape of the observation space
        shp = env.observation_space.shape
        shp = shp[0], shp[1], shp[2] * history_length
        # setup the buffer for frames to collect in
        self.frames = np.zeros(shp)
        # setup the new observation space based on the updated shape
        space = gym.spaces.Box(low=0, high=255, shape=shp, dtype=np.uint8)
        self.observation_space = space

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        # call the super method on this wrappers env
        frame = self.env.reset()
        # repeat the initial frame history length times to start the queue
        self.frames = np.repeat(frame, self.history_length, axis=2)
        # return the queue of initial frames
        return self.frames

    def step(self, action) -> tuple:
        """Perform an action and return  state, action, done flag, and info."""
        # perform the action and observe the new information
        frame, reward, done, info = self.env.step(action)
        # add the observation to the frame queue
        self.frames = np.concatenate((self.frames, frame), axis=2)
        # pop the last frame off the frame queue
        self.frames = self.frames[:, :, 1:]
        # return the frame queue, the reward, terminal flag, and info
        return self.frames, reward, done, info


# explicitly define the outward facing API of this module
__all__ = [FrameStackEnv.__name__]
