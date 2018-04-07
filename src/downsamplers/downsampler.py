"""An abstract base down-sampler class."""
import cv2
import numpy as np


class Downsampler(object):
    """A class for down-sampling images from RGB to Y."""

    def __init__(self, y: int, x: int, cut: list=[]) -> None:
        """
        Create a new down-sampler.

        Args:
            y: the coordinates of the Y padding (y_top, y_bottom)
            x: the coordinates of the X padding (x_left, x_right)
            cut: the shades of gray to cancel out to solid black

        Returns:
            None

        """
        self.y = y
        self.x = x
        self.cut = cut

    def __call__(self, frame: np.ndarray, image_size: tuple) -> np.ndarray:
        """
        Down-sample the given frame.

        Args:
            frame: the frame to down-sample
            image_size: the size of the image to return

        Returns:
            a down-sampled image reshaped to the given size i.e. a B&W version
            of frame resized to image_size and with certain colors removed

        """
        # convert the frame from RGB to gray scale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # crop the image to the playable space
        frame = frame[self.y[0]:-self.y[1], self.x[0]:-self.x[1]]
        # zero out specific colors
        frame[np.in1d(frame, self.cut).reshape(frame.shape)] = 0
        # resize the frame to the expected shape
        frame = cv2.resize(frame, image_size)

        return frame


# explicitly define the outward facing API of this module
__all__ = ['Downsampler']
