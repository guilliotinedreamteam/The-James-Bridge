import logging
import os
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                            ReduceLROnPlateau)
except ImportError:
    tf = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Phase 4: Training & Evaluation Pipeline.
    Manages the offline training loops, callbacks, and weight persistence for the Neurobridge models.
    """

    def __init__(self, model: "tf.keras.Model", checkpoint_dir: str = "checkpoints"):
        if tf is None:
            raise ImportError(
                "TensorFlow is missing. Cannot initialize training pipeline."
            )

        self.model = model
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            logger.info(f"Created checkpoint directory at {self.checkpoint_dir}")

    def get_callbacks(self, model_name: str) -> list:
        """
        Configures the standard Keras callbacks for robust training.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.keras")

        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
        ]
        return callbacks

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
        """
        Executes the offline training loop.
        """
        logger.info(
            f"Initiating training for {model_name}. Epochs: {epochs}, Batch Size: {batch_size}"
        )

        validation_data = (
            (x_val, y_val) if x_val is not None and y_val is not None else None
        )
        if validation_data is None:
            logger.warning(
                "No validation data provided. Model checkpointing may rely on training loss instead."
            )

        callbacks = self.get_callbacks(model_name)

        try:
            history = self.model.fit(
                x=x_train,
                y=y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
            )
            logger.info("Training cycle complete.")
            return history

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on unseen data and returns the metrics.
        """
        logger.info("Initiating model evaluation...")
        try:
            results = self.model.evaluate(x_test, y_test, return_dict=True)
            logger.info(f"Evaluation metrics: {results}")
            return results
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise e
