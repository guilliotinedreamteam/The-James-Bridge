"""
NeuroBridge Model — functional API

Provides standalone factory functions for building decoder models,
used by the test suite, training, and evaluation modules.
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import (LSTM, BatchNormalization, Bidirectional,
                                     Conv1D, Dense, Dropout, Input,
                                     SpatialDropout1D, TimeDistributed)

from neurobridge.config import Config

logger = logging.getLogger("neurobridge.model")


def build_neurobridge_decoder(
    timesteps: Optional[int] = None,
    features: Optional[int] = None,
    num_classes: Optional[int] = None,
    lstm_units: Optional[int] = None,
    dense_units: Optional[int] = None,
    dropout_rate: float = 0.3,
    cnn_filters: int = 64,
    cnn_kernel_size: int = 3,
) -> tf.keras.Model:
    """
    Build the offline CNN + Bidirectional LSTM hybrid decoder.

    Returns an un-compiled Keras functional model named
    ``neurobridge_decoder``.
    """
    timesteps = timesteps if timesteps is not None else Config.NUM_TIMESTEPS
    features = features if features is not None else Config.NUM_FEATURES
    num_classes = num_classes if num_classes is not None else Config.NUM_PHONEMES
    lstm_units = lstm_units if lstm_units is not None else Config.LSTM_UNITS
    dense_units = dense_units if dense_units is not None else Config.DENSE_UNITS

    inp = Input(shape=(timesteps, features), name="input")

    # --- CNN Feature Extractor ---
    # Extracts spatio-temporal motifs across the electrode array
    x = Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        padding="same",
        activation="relu",
        name="conv_1",
    )(inp)
    x = SpatialDropout1D(dropout_rate, name="sdrop_1")(x)
    x = Conv1D(
        filters=cnn_filters * 2,
        kernel_size=cnn_kernel_size,
        padding="same",
        activation="relu",
        name="conv_2",
    )(x)
    x = SpatialDropout1D(dropout_rate, name="sdrop_2")(x)

    # --- RNN Sequence Modeler ---
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bi_lstm")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(dropout_rate, name="drop_1")(x)

    x = TimeDistributed(Dense(dense_units, activation="relu"), name="td_dense")(x)
    x = Dropout(dropout_rate, name="drop_2")(x)

    # Output predicting the phoneme sequence
    out = TimeDistributed(Dense(num_classes, activation="softmax"), name="td_output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="neurobridge_decoder")
    return model


def build_realtime_decoder(
    features: Optional[int] = None,
    num_classes: Optional[int] = None,
    lstm_units: Optional[int] = None,
    dense_units: Optional[int] = None,
    cnn_filters: int = 64,
    cnn_kernel_size: int = 3,
) -> tf.keras.Model:
    """
    Build the real-time unidirectional CNN-LSTM hybrid decoder.

    Unlike the previous pure-LSTM which accepted 1 timestep, the CNN-LSTM
    requires a temporal buffer to perform convolutions (e.g. 10 timesteps).

    Returns an un-compiled Keras functional model named
    ``neurobridge_realtime_decoder``.
    """
    features = features if features is not None else Config.NUM_FEATURES
    num_classes = num_classes if num_classes is not None else Config.NUM_PHONEMES
    lstm_units = lstm_units if lstm_units is not None else Config.LSTM_UNITS
    dense_units = dense_units if dense_units is not None else Config.DENSE_UNITS

    # We must accept a temporal buffer (e.g., 10 frames) instead of 1 frame
    # We leave the timestep dimension as `None` to be flexible
    inp = Input(shape=(None, features), name="rt_input")

    # --- CNN Feature Extractor ---
    # MUST mirror the offline model exactly so weights can be loaded
    x = Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        padding="same",
        activation="relu",
        name="conv_1",
    )(inp)
    x = Conv1D(
        filters=cnn_filters * 2,
        kernel_size=cnn_kernel_size,
        padding="same",
        activation="relu",
        name="conv_2",
    )(x)

    # --- RNN Sequence Modeler ---
    x = LSTM(lstm_units, return_sequences=True, name="rt_lstm")(x)
    x = BatchNormalization(name="rt_bn")(x)

    x = TimeDistributed(Dense(dense_units, activation="relu"), name="rt_dense")(x)
    x = TimeDistributed(Dense(num_classes, activation="softmax"), name="rt_output")(x)

    # For real-time, we typically care about the MOST RECENT prediction in the buffer.
    # We slice out the last timestep's prediction.
    # The output of TimeDistributed is (batch, condensed_timesteps, num_classes)
    # We take [:, -1, :] to get (batch, num_classes)
    out = x[:, -1, :]

    model = tf.keras.Model(inputs=inp, outputs=out, name="neurobridge_realtime_decoder")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: Optional[float] = None,
) -> None:
    """
    Compile a Keras model in-place with Adam + categorical cross-entropy.
    """
    learning_rate = learning_rate if learning_rate is not None else Config.LEARNING_RATE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
