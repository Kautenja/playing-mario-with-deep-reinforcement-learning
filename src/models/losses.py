"""Loss functions for models in the project."""
import tensorflow as tf
from keras import backend as K


def huber_loss(y, y_pred, delta: int=1.0):
    """
    Return the Huber loss between tensors.

    Reference:
        https://en.wikipedia.org/wiki/Huber_loss
        https://web.stanford.edu/class/cs20si/2017/lectures/slides_03.pdf
        https://keras.io/backend/

    Args:
        y: ground truth y labels
        y_pred: predicted y labels
        delta: the

    Returns:
        a scalar loss between the ground truth and predicted labels

    """
    residual = K.abs(y_pred - y)
    condition = K.less_equal(residual, delta)
    then = 0.5 * K.square(residual)
    otherwise = delta * residual - 0.5 * K.square(delta)
    return K.switch(condition, then, otherwise)


# explicitly define the outward facing API of this module
__all__ = ['huber_loss']
