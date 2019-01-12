"""A method to count the weights in a model."""
from keras import backend as K


def non_trainable_weights(model) -> int:
    """
    Return the number of non-trainable parameters in a model.

    Args:
        model: the model to count the non-trainable parameters in

    Returns:
        the number of non-trainable parameters in the model

    """
    weights = set(model.non_trainable_weights)
    return sum(K.count_params(p) for p in weights)


def trainable_weights(model) -> int:
    """
    Return the number of trainable parameters in a model.

    Args:
        model: the model to count the trainable parameters in

    Returns:
        the number of trainable parameters in the model

    """
    weights = set(model.trainable_weights)
    return sum(K.count_params(p) for p in weights)


def count_weights(model) -> tuple:
    """
    Return the number of weights in the given model.

    Args:
        model: the model to count the number of weights in

    Returns:
        a tuple with two items:
        - the number of trainable weights
        - the number of non-trainable weights

    """
    return trainable_weights(model), non_trainable_weights(model)


# explicitly define the outward facing API of the module
__all__ = [
    count_weights.__name__,
    non_trainable_weights.__name__,
    trainable_weights.__name__,
]
