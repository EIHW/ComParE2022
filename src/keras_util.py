import tensorflow as tf
from keras import backend as K

# Deep learning
# Keras-related imports

import config
import config_keras  # local module

K.set_image_data_format("channels_first")
import os
from datetime import datetime

from keras.callbacks import ModelCheckpoint

# TF2 changes:
from mozbnn_model import build_model


def train_model(
    X_train, y_train, X_val=None, y_val=None, class_weight=None, start_from=None
):

    y_train = tf.keras.utils.to_categorical(y_train)
    if y_val is not None:
        y_val = tf.keras.utils.to_categorical(y_val)

    if start_from is not None:
        model = load_model(start_from)
        print("Starting from model", start_from)
    else:
        model = build_model()

        # if checkpoint_name is not None:
        # 	os.path.join(os.path.pardir, 'models', 'keras', checkpoint_name)

    if X_val is None:
        metric = "accuracy"
        val_data = None
    else:
        metric = "val_accuracy"
        val_data = (X_val, y_val)

    model_name = (
        "Win_" + str(config.win_size) + "_Stride_" + str(config.step_size) + "_"
    )
    model_name = (
        model_name
        + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        + "-e{epoch:02d}"
        + metric
        + "{"
        + metric
        + ":.4f}.hdf5"
    )
    # model_name = model_name + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '-e{epoch:02d}.hdf5'
    checkpoint_filepath = os.path.join(config.model_dir, "keras", model_name)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor=metric,
        mode="max",
        save_best_only=False,
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=config_keras.batch_size,
        epochs=config_keras.epochs,
        verbose=1,
        validation_data=val_data,
        shuffle=True,
        class_weight=class_weight,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        callbacks=[model_checkpoint_callback],
    )

    return model, model_name


def evaluate_model(model, X_test, y_test, n_samples):
    all_y_pred = []
    for n in range(n_samples):
        all_y_pred.append(model.predict(X_test))
    return all_y_pred


def load_model(filepath):
    model = tf.keras.models.load_model(
        filepath, custom_objects={"dropout": config_keras.dropout}
    )
    return model
