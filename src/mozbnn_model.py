from keras import backend as K
from keras.layers import Conv2D, Dense, Flatten, Lambda, MaxPooling2D
from keras.models import Sequential
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adadelta

import config
import config_keras

K.set_image_data_format("channels_first")

# Adadelta optimizer configuration
learning_rate = config_keras.learning_rate
rho = config_keras.rho
epsilon = config_keras.epsilon

# BNN weight regularisation:
tau = config_keras.tau  # config_keras.tau
dropout = config_keras.dropout  # config_keras.dropout

# Settings for mosquito event detection:
lengthscale = config_keras.lengthscale


def build_model():
    input_shape = (1, config.win_size, config.n_feat)

    reg = (
        lengthscale ** 2
        * (1 - dropout)
        / (2.0 * 100000 * tau)  # Warning: normalising constant changing
    )  # reg = lengthscale**2 * (1 - dropout) / (2. * len(X_train) * tau)

    # Initialise optimiser for consistent results across Keras/TF versions
    opt = Adadelta(learning_rate=learning_rate, rho=rho, epsilon=epsilon)

    model = Sequential()
    n_dense = 128
    nb_classes = 2
    nb_conv_filters = 32
    nb_conv_filters_2 = 64

    model.add(
        Conv2D(
            nb_conv_filters,
            kernel_size=(3, 3),
            activation="relu",
            padding="valid",
            strides=1,
            input_shape=input_shape,
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Lambda(lambda x: K.dropout(x, level=dropout)))

    model.add(
        Conv2D(
            nb_conv_filters_2, kernel_size=(3, 3), activation="relu", padding="valid"
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Lambda(lambda x: K.dropout(x, level=dropout)))

    model.add(
        Conv2D(
            nb_conv_filters_2, kernel_size=(3, 3), activation="relu", padding="valid"
        )
    )
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Lambda(lambda x: K.dropout(x, level=dropout)))

    # # model.add(Dropout(0.2))
    model.add(
        Conv2D(
            nb_conv_filters_2, kernel_size=(3, 3), activation="relu", padding="valid"
        )
    )
    model.add(Lambda(lambda x: K.dropout(x, level=dropout)))

    model.add(Flatten())

    # Shared between MLP and CNN:
    model.add(Dense(n_dense, activation="relu"))
    model.add(Lambda(lambda x: K.dropout(x, level=dropout)))

    model.add(Dense(nb_classes, activation="softmax", kernel_regularizer=l2(reg)))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model
