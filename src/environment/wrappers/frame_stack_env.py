"""An environment wrapper to stack observations into a tensor."""
from collections import deque
import numpy as np
import gym


# class FrameStackEnv(gym.Wrapper):
#     """An environment wrapper to stack observations into a tensor."""

#     def __init__(self, env, history_length: int=4) -> None:
#         """
#         Initialize a new frame stacking environment.

#         Args:
#             env: the environment to stack frames of
#             history_length: the number of frames to stack

#         """
#         gym.Wrapper.__init__(self, env)
#         self.history_length = history_length
#         # unwrap the new shape of the observation space
#         shp = env.observation_space.shape
#         shp = shp[0], shp[1], shp[2] * history_length
#         # setup the buffer for frames to collect in
#         self.frames = np.zeros(shp)
#         # setup the new observation space based on the updated shape
#         space = gym.spaces.Box(low=0, high=255, shape=shp, dtype=np.uint8)
#         self.observation_space = space

#     def reset(self) -> np.ndarray:
#         """Reset the environment."""
#         # call the super method on this wrappers env
#         frame = self.env.reset()
#         # repeat the initial frame history length times to start the queue
#         self.frames = np.repeat(frame, self.history_length, axis=2)
#         # return the queue of initial frames
#         return self.frames

#     def step(self, action) -> tuple:
#         """Perform an action and return  state, action, done flag, and info."""
#         # perform the action and observe the new information
#         frame, reward, done, info = self.env.step(action)
#         # add the observation to the frame queue
#         self.frames = np.concatenate((self.frames, frame), axis=2)
#         # pop the last frame off the frame queue
#         self.frames = self.frames[:, :, 1:]
#         # return the frame queue, the reward, terminal flag, and info
#         return self.frames, reward, done, info


class FrameStackEnv(gym.Wrapper):
    """An environment wrapper to stack observations into a tensor."""

    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=np.uint8
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """A memory efficient buffer for frame tensors."""

    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers. This object should only be
        converted to numpy array before being passed to the model. You'd not
        believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


# explicitly define the outward facing API of this module
__all__ = [FrameStackEnv.__name__]
