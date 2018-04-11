"""Classes for down-sampling images from OpenAI Gym Environments."""
from .downsampler import Downsampler


# A Pong down-sampler with a very tight crop to the playable areas
downsample_pong = Downsampler(y=(34, 16), x=(15, 15), cut=[107, 87])

# A Breakout down-sampler with a modest crop, neither tight nor heavily padded
downsample_breakout = Downsampler(y=(32, 14), x=(8, 8), cut=[142])

# A Space Invaders down-sampler with the best crop possible
downsample_space_invaders = Downsampler(y=(0, 15), x=(0, 1), cut=[])


# explicitly define the outward facing API for this package
__all__ = [
    'Downsampler',
    'downsample_pong',
    'downsample_breakout',
    'downsample_space_invaders'
]
