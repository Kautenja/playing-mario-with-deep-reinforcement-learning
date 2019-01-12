"""A wrapper for training procedures."""
from ..utils import seed
from ..utils import set_gpu


def stochastic(seed_: int, gpu: int) -> 'Callable':
    """
    Build a stochastic wrapper.

    Args:
        seed_: the random number seed to use
        gpu: the gpu to select to the operation

    Returns:
        a callable wrapper for a stochastic method

    """
    def build_stochastic(method: 'Callable') -> 'Callable':
        """
        Build a stochastic wrapper.

        Args:
            method: the method to wrap

        Returns:
            a callable method wrapped by training boilerplate

        """
        def do_method(*args, **kwargs) -> any:
            """Call a stochastic method with boilerplate functionality."""
            # seed the random number generators
            seed(seed_)
            # set the GPU if specified
            set_gpu(gpu)
            # call the method
            try:
                return method(*args, **kwargs)
            except KeyboardInterrupt:
                print('\ncaught Keyboard Interrupt. terminating...')

        # reassign the name of the method
        do_method.__name__ = method.__name__

        return do_method

    return build_stochastic


# explicitly define the outward facing API of this module
__all__ = [stochastic.__name__]
