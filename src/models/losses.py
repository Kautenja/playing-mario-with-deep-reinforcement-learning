"""Loss functions for models in the project."""
from keras import backend as K


def huber_loss(y, y_pred, delta: float=1.0):
    """
    Return the Huber loss between tensors.

    Reference:
        https://en.wikipedia.org/wiki/Huber_loss
        https://web.stanford.edu/class/cs20si/2017/lectures/slides_03.pdf
        https://keras.io/backend/

    Args:
        y: ground truth y labels
        y_pred: predicted y labels
        delta: the separating constant between MSE and MAE

    Returns:
        a scalar loss between the ground truth and predicted labels

    """
    # calculate the residuals
    residual = K.abs(y_pred - y)
    # determine the result of the logical comparison to delta
    condition = K.less_equal(residual, delta)
    # calculate the two possible returns (MSE and MAE)
    then_this = 0.5 * K.square(residual)
    else_this = delta * residual - 0.5 * K.square(delta)
    # use the condition to determine the resulting tensor
    return K.switch(condition, then_this, else_this)


# explicitly define the outward facing API of this module
__all__ = [huber_loss.__name__]
