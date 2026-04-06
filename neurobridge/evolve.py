import logging
import random
import numpy as np
from typing import Dict, Any

from neurobridge.model.decoder import NeurobridgeDecoder
from neurobridge.training.trainer import ModelTrainer
from neurobridge.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evolver:
    """
    Recursive Hyperparameter Evolution for Neurobridge BCI Models.
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, epochs_per_gen: int = 2):
        self.x_data = x_data
        self.y_data = y_data
        self.epochs_per_gen = epochs_per_gen
        self.best_config = None
        self.best_accuracy = -1.0

    def _evaluate_config(self, params: Dict[str, Any]) -> float:
        """Evaluates a single hyperparameter configuration."""
        timesteps = self.x_data.shape[1]
        channels = self.x_data.shape[2]

        decoder = NeurobridgeDecoder(
            timesteps=timesteps,
            channels=channels,
            lstm_units=params['lstm_units'],
            dense_units=params['dense_units'],
            dropout_rate=params['dropout_rate']
        )
        model = decoder.build_offline_decoder()

        # Override learning rate if needed, here we'll use Adam's default or whatever is in model.compile
        # but to be strict with the task "mutate hyperparameters (learning rate, dropout, units)"
        # We need to compile model with new learning rate
        import tensorflow as tf
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        trainer = ModelTrainer(model=model, checkpoint_dir="evolution_checkpoints")
        try:
            history = trainer.train(
                x_train=self.x_data,
                y_train=self.y_data,
                epochs=self.epochs_per_gen,
                batch_size=8,
                model_name=f"evolve_gen"
            )
            return history.history['accuracy'][-1]
        except Exception as e:
            logger.error(f"Evaluation failed for config {params}: {e}")
            return 0.0

    def _evolve_recursive(self, current_params: Dict[str, Any], generations_left: int) -> Dict[str, Any]:
        """Recursively mutates and evaluates hyperparameters."""
        if generations_left == 0:
            return self.best_config

        logger.info(f"--- Generation {101 - generations_left}/100 ---")

        # Mutate params
        # Learning rate: +/- 20%
        # Dropout: +/- 0.05
        # LSTM units: +/- 16
        # Dense units: +/- 16
        mutated_params = current_params.copy()

        # Small chance to randomly explore, mostly exploit + mutate
        mutated_params['learning_rate'] = max(1e-5, min(0.1, mutated_params['learning_rate'] * random.uniform(0.8, 1.2)))
        mutated_params['dropout_rate'] = max(0.0, min(0.9, mutated_params['dropout_rate'] + random.uniform(-0.05, 0.05)))
        mutated_params['lstm_units'] = max(32, mutated_params['lstm_units'] + random.choice([-16, 0, 16]))
        mutated_params['dense_units'] = max(32, mutated_params['dense_units'] + random.choice([-16, 0, 16]))

        logger.info(f"Evaluating Config: {mutated_params}")
        accuracy = self._evaluate_config(mutated_params)
        logger.info(f"Accuracy: {accuracy:.4f}")

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_config = mutated_params.copy()
            logger.info(f"New Best! Accuracy: {self.best_accuracy:.4f} Config: {self.best_config}")

            # Save to Config
            Config.LEARNING_RATE = self.best_config['learning_rate']
            Config.LSTM_UNITS = self.best_config['lstm_units']
            Config.DENSE_UNITS = self.best_config['dense_units']
            # Dropout doesn't have a direct slot in Config but we'll save it if needed.

        return self._evolve_recursive(self.best_config, generations_left - 1)

    def run(self, generations: int = 100):
        initial_params = {
            'learning_rate': Config.LEARNING_RATE,
            'dropout_rate': 0.3,
            'lstm_units': Config.LSTM_UNITS,
            'dense_units': Config.DENSE_UNITS
        }

        logger.info(f"Starting evolution for {generations} generations...")
        self.best_config = initial_params.copy()
        self.best_accuracy = self._evaluate_config(initial_params)

        logger.info(f"Initial accuracy: {self.best_accuracy:.4f}")

        final_config = self._evolve_recursive(initial_params, generations)
        logger.info(f"Evolution complete! Best accuracy: {self.best_accuracy:.4f}")
        logger.info(f"Best config: {final_config}")
        return final_config
