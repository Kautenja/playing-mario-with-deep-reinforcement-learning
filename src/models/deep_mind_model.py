"""The Deep Mind Convolutional Neural Network (CNN) model."""
from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Multiply
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop


def build_deep_mind_model(
    image_size: tuple=(84, 84),
    num_frames: int=4,
    num_actions: int=6,
    learning_rate: float=1e-5
) -> Model:
    """
    Build and return the Deep Mind model for the given domain parameters.

    Notes:
        Color Space: this CNN expects single channel images (B&W)

    Args:
        input_shape: the shape of the image states for the model
                     Atari games are (192, 160), but DeepMind reduced the
                     size to (84, 84) to reduce computational load
        num_frames: the number of frames being stacked together
                    DeepMind uses 4 frames in their original implementation
        num_actions: the output shape for the model, this represents the
                     number of discrete actions available to a game
        learning_rate: the learning rate for the optimization method for the
                       network

    Returns:
        a blank DeepMind CNN for image classification in a reinforcement agent

    """
    # build the CNN using the functional API
    cnn_input = Input((*image_size, num_frames), name='cnn')
    cnn = Lambda(lambda x: x / 255.0)(cnn_input)
    cnn = Conv2D(32, (8, 8), strides=(4,4), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(64, (4, 4), strides=(2,2), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(64, (3, 3), strides=(1,1), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(512)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dense(num_actions)(cnn)
    # build the mask using the functional API
    mask_input = Input((num_actions,), name='mask')
    # put the two pieces of the graph together
    output = Multiply()([cnn, mask_input])

    # build the model
    model = Model(input=[cnn_input, mask_input], output=output)
    # compile the model with the default loss and optimization technique
    # TODO: parameterize optimizer, is learning rate necessary?
    # model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(loss='mse', optimizer=optimizer)

    return model


# explicitly define the outward facing API of this module
__all__ = ['build_deep_mind_model']
