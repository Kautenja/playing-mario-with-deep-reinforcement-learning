"""An environment wrapper for down-sampling frames from RGB to smaller B&W."""
import gym
import cv2
import numpy as np


class DownsampleEnv(gym.ObservationWrapper):
    """Downsample frames to a B&W with a cropped size and colors removed."""

    # metadata for the down-samplers for each game
    metadata = {
        'Pong': {
            'y': (34, 16),
            'x': (15, 15),
            'cut': [107, 87]
        },
        'Breakout': {
            'y': (32, 14),
            'x': (8, 8),
            'cut': [142]
        },
        'SpaceInvaders': {
            'y': (0, 15),
            'x': (0, 1),
            'cut': []
        },
    }

    def __init__(self, env, image_size: tuple, y: int, x: int, cut: list=[]):
        """
        Create a new down-sampler.

        Args:
            image_size: the size to output frames as
            y: the coordinates of the Y padding (y_top, y_bottom)
            x: the coordinates of the X padding (x_left, x_right)
            cut: the shades of gray to cancel out to solid black

        Returns:
            None

        """
        gym.ObservationWrapper.__init__(self, env)
        self.image_size = image_size
        self.y = y
        self.x = x
        self.cut = cut
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(image_size[1], image_size[0], 1),
            dtype=np.uint8
        )

    def observation(self, frame):
        # convert the frame from RGB to gray scale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # crop the image to the playable space
        frame = frame[self.y[0]:-self.y[1], self.x[0]:-self.x[1]]
        # zero out specific colors
        frame[np.in1d(frame, self.cut).reshape(frame.shape)] = 0
        # resize the frame to the expected shape
        frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)

        return frame[:, :, np.newaxis]
        # return frame[:, :, None]


# explicitly define the outward facing API of this module
__all__ = ['DownsampleEnvs']
