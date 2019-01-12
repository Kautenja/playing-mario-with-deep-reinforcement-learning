"""A method to determine model names of GPUs on a machine."""
from tensorflow.python.client import device_lib


def get_device_model(device) -> str:
    """
    Return the model of a TensorFlow device.

    Args:
        device: the device to get the model name of

    Returns:
        a string describing the model name of the device

    """
    # get the physical description of the device, extract the name, and remove
    # extra text
    return device.physical_device_desc.split(', name: ')[-1].split(',')[0]


def get_gpu_models() -> list:
    """Return a list of the available GPUs by model name."""
    # get the devices on the machine
    devices = device_lib.list_local_devices()
    # return a list of the devices by model name
    return [get_device_model(d) for d in devices if d.device_type == 'GPU']


# explicitly define the outward facing API of this module
__all__ = [
    get_gpu_models.__name__,
]
