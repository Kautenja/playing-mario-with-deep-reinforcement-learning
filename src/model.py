"""The Alpha Go Convolutional Neural Network model."""


def build_model(input_shape: tuple=(224, 256, 3), output_shape: int=6):
    """
    Build and return the Alpha Go model for the given domain parameters.

    Args:
        input_shape: the shape of the image states for the model
        output_shape: the output shape for the model (predicted classes)

    Returns: a compiled keras model for image classification
    """
    # lazy import inside this method to reduce the overhead that importing
    # keras introduces by default
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv2D
    from keras.optimizers import Adam
    # build the model for image classification fitting the given parameters
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4,4), padding='same', activation='relu', input_shape=input_shape),
        Conv2D(64, (4, 4), strides=(2,2), padding='same', activation='relu'),
        Conv2D(64, (3, 3), strides=(1,1), padding='same', activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(output_shape)
    ])
    # compile the model with the default loss and optimization technique
    model.compile(loss='mse', optimizer=Adam(lr=1e-6))

    return model
