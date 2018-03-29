"""The Deep Mind Convolutional Neural Network (CNN) model."""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam


# TODO: explore conflicting information about the input_shape. DeepMind seems
#       to have used grayscale images of size 84. i.e. shape (84, 84, 1), but
#       some sources use a different shape with RGB color space (224, 256, 3)
#       black and white will certainly reduce the computational complexity
#       but will that loss of information affect performance?


def build_model(input_shape: tuple=(224, 256, 3), output_dim: int=6):
    """
    Build and return the Deep Mind model for the given domain parameters.

    Args:
        input_shape: the shape of the image states for the model
        output_dim: the output shape for the model (predicted classes)

    Returns:
        a blank CNN for image classification

    """
    # build the model for image classification fitting the given parameters
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4,4), padding='same', activation='relu', input_shape=input_shape),
        Conv2D(64, (4, 4), strides=(2,2), padding='same', activation='relu'),
        Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    # compile the model with the default loss and optimization technique
    model.compile(loss='mse', optimizer=Adam(lr=1e-6))

    return model


__all__ = ['build_model']
