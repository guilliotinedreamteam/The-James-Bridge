"""
NeuroBridge Training Pipeline

Provides the training loop with proper callbacks, logging,
and model checkpointing.
"""

import logging
from pathlib import Path
from typing import Optional

import tensorflow as tf

from neurobridge.config import Config
from neurobridge.data import create_data_generator
from neurobridge.model import build_neurobridge_decoder, compile_model

logger = logging.getLogger("neurobridge.train")


def train_model(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    train_samples: Optional[int] = None,
    model: Optional[tf.keras.Model] = None,
    save_path: Optional[Path] = None,
) -> tf.keras.callbacks.History:
    """
    Train the NeuroBridge decoder model.

    Builds (or uses a provided) model, trains it on mock data,
    and saves the result.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        train_samples: Total conceptual training samples per epoch.
        model: Pre-built model (if None, builds a new one).
        save_path: Where to save the trained model.

    Returns:
        Keras History object containing training metrics.
    """
    epochs = epochs or Config.EPOCHS
    batch_size = batch_size or Config.BATCH_SIZE
    train_samples = train_samples or Config.TRAIN_SAMPLES
    save_path = save_path or Config.model_save_path()

    steps_per_epoch = max(1, train_samples // batch_size)

    # Build and compile model if not provided
    if model is None:
        logger.info("Building new decoder model...")
        model = build_neurobridge_decoder()
        compile_model(model)
    elif not model.optimizer:
        compile_model(model)

    model.summary(print_fn=logger.info)

    # Callbacks for production-quality training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info(
        "Starting training: epochs=%d, batch_size=%d, steps_per_epoch=%d",
        epochs,
        batch_size,
        steps_per_epoch,
    )

    # Train
    generator = create_data_generator(batch_size)
    history = model.fit(
        generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    logger.info("Saving final model to: %s", save_path)
    model.save(str(save_path))
    logger.info("Training complete. Model saved.")

    return history
