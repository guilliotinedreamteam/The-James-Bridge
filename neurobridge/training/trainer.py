import logging
import os
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf

# 1. Import your new custom loss function
from neurobridge.model.losses import CategoricalFocalLoss

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training lifecycle, callbacks, and weight persistence."""

    def __init__(self, model: tf.keras.Model, learning_rate: float = 1e-4):
        self.model = model
        self.learning_rate = learning_rate

        # 2. Replaced 'categorical_crossentropy' with CategoricalFocalLoss
        logger.info(f"Compiling model with Adam (lr={learning_rate}) and Focal Loss.")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=CategoricalFocalLoss(gamma=2.0, alpha=0.25),
            metrics=["accuracy"],
        )

    # 3. Inject the sample weighting logic
    def compute_temporal_sample_weights(self, y_train: np.ndarray) -> np.ndarray:
        """
        Computes 2D sample weights (batch_size, timesteps) to counteract class imbalance.
        y_train shape: (batch, timesteps, num_classes) - expected one-hot
        """
        logger.info("Computing temporal sample weights to prevent model collapse...")
        
        # Flatten to count class frequencies
        y_flat = np.argmax(y_train, axis=-1).flatten()
        classes, counts = np.unique(y_flat, return_counts=True)
        
        # Inverse frequency weighting
        total_samples = len(y_flat)
        class_weights = {cls: total_samples / (len(classes) * count) for cls, count in zip(classes, counts)}
        
        # Cap the maximum weight to prevent gradient explosions from extremely rare phonemes
        max_weight = 10.0
        class_weights = {k: min(v, max_weight) for k, v in class_weights.items()}
        
        # Create the (batch, timesteps) weight matrix
        y_labels = np.argmax(y_train, axis=-1)
        sample_weights = np.vectorize(class_weights.get)(y_labels)
        
        logger.info(f"Class weight spread: Min={min(class_weights.values()):.2f}, Max={max(class_weights.values()):.2f}")
        return sample_weights

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        model_name: str = "neurobridge_offline",
    ) -> "tf.keras.callbacks.History":
        """Executes the training loop with early stopping and checkpointing."""
        
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/{model_name}_best.keras"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor="val_loss" if x_val is not None else "loss",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if x_val is not None else "loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if x_val is not None else "loss",
                factor=0.5,
                patience=5,
                verbose=1,
            ),
        ]

        # 4. Generate sample weights for the training data
        sample_weights = self.compute_temporal_sample_weights(y_train)

        validation_data = None
        if x_val is not None and y_val is not None:
            # 5. Generate sample weights for the validation data tuple
            val_weights = self.compute_temporal_sample_weights(y_val)
            validation_data = (x_val, y_val, val_weights)

        logger.info(f"Starting training for {epochs} epochs (batch={batch_size})")

        history = self.model.fit(
            x=x_train,
            y=y_train,
            sample_weight=sample_weights,  # 6. Apply weights to the fit function
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        final_path = f"checkpoints/{model_name}_final.keras"
        self.model.save(final_path)
        logger.info(f"Training complete. Final model saved to {final_path}")

        return history
