"""An implementation of the model for an A3C agent."""
from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras import backend as K


def policy_loss(policy, action, advantage):
    log_pi = K.log(K.sum(policy * action, axis=1) + K.epsilon())
    return -log_pi * advantage


def value_loss(advantage, weight=0.5):
    return weight * K.square(advantage)


def entropy_loss(policy, weight=0.01):
    return weight * K.sum(policy * K.log(policy + K.epsilon()), axis=1)


def build_a3c_model(
    image_size: tuple=(84, 84),
    num_frames: int=4,
    num_actions: int=6,
    optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
    value_weight=0.5,
    entropy_weight=0.01
) -> Model:
    """
    Build and return the A3C model for the given domain parameters.

    Notes:
        Color Space: this CNN expects single channel images (B&W)

    Args:
        image_size: the shape of the image states for the model
        num_frames: the number of frames being stacked together
        num_actions: the number of actions in the environment (output shape)
        optimizer: the optimizer for reducing error from batches
        value_weight: the weight for the value loss
        entropy_weight: the weight for the entropy regularization loss

    Returns:
        a blank A3C CNN for image classification in a reinforcement agent

    """
    # build the CNN using the functional API
    cnn_input = Input((*image_size, num_frames), name='cnn')
    cnn = Lambda(lambda x: x / 255.0)(cnn_input)
    cnn = Conv2D(32, (8, 8), strides=(4, 4), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(512)(cnn)
    cnn = Activation('relu')(cnn)

    # build an output layer for the probabilities from the policy pi(s)
    output_policy = Dense(num_actions)(cnn)
    output_policy = Activation('softmax')(output_policy)

    # build an output for the value function V(s)
    output_value = Dense(1)(cnn)
    output_value = Activation('linear')(output_value)

    # build batch inputs relating to the loss function
    s_input = Input((*image_size, num_frames), name='s')
    a_input = Input((num_actions,), name='a')
    r_input = Input((1,), name='r')

    # calculate the advantage
    advantage = r_input - output_value
    # calculate component losses for policy, value, and entropy regularization
    L_policy = policy_loss(output_policy, a_input, advantage)
    L_value = value_loss(advantage, weight=value_weight)
    L_entropy = entropy_loss(output_policy, weight=entropy_weight)

    # calculate the total loss and
    L_total = K.mean(L_policy + L_value + L_entropy)
    loss = lambda y_pred, y: L_total

    # build the model
    model = Model(
        inputs=[cnn_input, s_input, a_input, r_input],
        outputs=[output_policy, output_value]
    )
    # compile the model with the default loss and optimization technique
    model.compile(loss=loss, optimizer=optimizer)

    return model


# explicitly define the outward facing API of this module
__all__ = ['build_a3c_model']
