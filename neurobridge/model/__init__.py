"""
NeuroBridge Model — functional API

Provides standalone factory functions for building decoder models,
used by the test suite, training, and evaluation modules.
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional,
    TimeDistributed, BatchNormalization, Dropout, Reshape,
)


from neurobridge.config import Config

logger = logging.getLogger("neurobridge.model")


def build_neurobridge_decoder(
    timesteps: int = None,
    features: int = None,
    num_classes: int = None,
    lstm_units: int = None,
    dense_units: int = None,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Build the offline bidirectional LSTM decoder.

    Returns an un-compiled Keras functional model named
    ``neurobridge_decoder``.
    """
    timesteps = timesteps if timesteps is not None else Config.NUM_TIMESTEPS
    features = features if features is not None else Config.NUM_FEATURES
    num_classes = num_classes if num_classes is not None else Config.NUM_PHONEMES
    lstm_units = lstm_units if lstm_units is not None else Config.LSTM_UNITS
    dense_units = dense_units if dense_units is not None else Config.DENSE_UNITS

    inp = Input(shape=(timesteps, features), name="input")
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bi_lstm")(inp)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(dropout_rate, name="drop_1")(x)
    x = TimeDistributed(Dense(dense_units, activation="relu"), name="td_dense")(x)
    x = Dropout(dropout_rate, name="drop_2")(x)
    out = TimeDistributed(Dense(num_classes, activation="softmax"), name="td_output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="neurobridge_decoder")
    return model


def build_realtime_decoder(
    features: int = None,
    num_classes: int = None,
    lstm_units: int = None,
    dense_units: int = None,
) -> tf.keras.Model:
    """
    Build the real-time unidirectional LSTM decoder.

    Returns an un-compiled Keras functional model named
    ``neurobridge_realtime_decoder``.
    """
    features = features if features is not None else Config.NUM_FEATURES
    num_classes = num_classes if num_classes is not None else Config.NUM_PHONEMES
    lstm_units = lstm_units if lstm_units is not None else Config.LSTM_UNITS
    dense_units = dense_units if dense_units is not None else Config.DENSE_UNITS

    inp = Input(shape=(1, features), name="rt_input")
    x = LSTM(lstm_units, return_sequences=True, name="rt_lstm")(inp)
    x = BatchNormalization(name="rt_bn")(x)
    x = TimeDistributed(Dense(dense_units, activation="relu"), name="rt_dense")(x)
    x = TimeDistributed(Dense(num_classes, activation="softmax"), name="rt_output")(x)
    out = Reshape((num_classes,), name="rt_squeeze")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="neurobridge_realtime_decoder")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = None,
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
