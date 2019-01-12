"""A method to set the GPU to use with Keras."""
import os


def set_gpu(gpu) -> None:
    """
    Set the GPU to use with Keras to a specific value.

    Args:
        gpu: the GPU to use with Keras

    Returns:
        None

    """
    if gpu == '' or gpu >= 0:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


# explicitly define the outward facing API of this module
__all__ = [set_gpu.__name__]
