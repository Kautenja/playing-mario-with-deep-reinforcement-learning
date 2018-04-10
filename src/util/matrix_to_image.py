"""A method for converting NumPy matrices into proprietary images."""
import numpy as np
from PIL import Image


def matrix_to_image(image: np.ndarray, channel_range: tuple=(0, 255)) -> Image:
    """
    Convert the input matrix to an image.
    Args:
        image: the matrix of shape [height, width, channel] to convert
        channel_range: the range to clip the channel values to (inclusive)
    Returns:
        an image from the pixels in the image array
    """
    # clip the values in the image to the boundary [0, 255]. This is the
    # legal range for channel values. Image uses a method called 'to bytes'
    # to compress the input array into a simpler binary representation for
    # graphics processing. As such, convert the type to a single byte to
    # satisfy this constraint.
    image = np.clip(image, *channel_range).astype(np.uint8)

    return Image.fromarray(image)


# explicitly define the outward facing API of this module
__all__ = ['matrix_to_image']
