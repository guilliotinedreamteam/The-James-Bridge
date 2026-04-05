"""
NeuroBridge Model Evaluation

Provides functions to evaluate trained model performance
using phoneme-level accuracy and classification reports.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from neurobridge.config import Config
from neurobridge.data import generate_mock_ecog_data, generate_mock_phoneme_labels

logger = logging.getLogger("neurobridge.evaluate")


def load_model(model_path: Optional[Path] = None) -> tf.keras.Model:
    """
    Load a saved Keras model.

    Args:
        model_path: Path to the saved model file.

    Returns:
        The loaded Keras model.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    model_path = model_path or Config.model_save_path()

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run training first with: python neurobridge.py train"
        )

    logger.info("Loading model from: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded successfully.")
    return model


def evaluate_model(
    model: Optional[tf.keras.Model] = None,
    test_ecog: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    num_test_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate the decoder model on test data.

    If no test data is provided, generates mock test data.

    Args:
        model: Trained Keras model (loads from disk if None).
        test_ecog: Test ECoG data, shape (samples, timesteps, features).
        test_labels: Test labels (one-hot), shape (samples, timesteps, classes).
        num_test_samples: Number of mock samples if generating test data.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Load model if not provided
    if model is None:
        model = load_model()

    # Generate mock test data if not provided
    if test_ecog is None:
        logger.info("Generating %d mock test samples...", num_test_samples)
        test_ecog = generate_mock_ecog_data(num_test_samples)
        test_labels = generate_mock_phoneme_labels(num_test_samples)

    if test_labels is None:
        raise ValueError("test_labels must be provided when test_ecog is provided")

    logger.info(
        "Test data shapes — ECoG: %s, Labels: %s",
        test_ecog.shape, test_labels.shape,
    )

    # Predict
    logger.info("Running prediction on test data...")
    predicted_probs = model.predict(test_ecog, verbose=0)
    logger.info("Predictions shape: %s", predicted_probs.shape)

    # Convert to discrete phoneme IDs
    predicted_ids = np.argmax(predicted_probs, axis=-1)
    true_ids = np.argmax(test_labels, axis=-1)

    # Phoneme-level (frame-wise) accuracy
    matches = (predicted_ids == true_ids).astype(np.float32)
    accuracy = float(np.mean(matches))

    # Per-sample accuracy
    per_sample_acc = np.mean(matches, axis=1)
    best_sample_acc = float(np.max(per_sample_acc))
    worst_sample_acc = float(np.min(per_sample_acc))

    metrics = {
        "phoneme_accuracy": accuracy,
        "best_sample_accuracy": best_sample_acc,
        "worst_sample_accuracy": worst_sample_acc,
        "num_test_samples": test_ecog.shape[0],
        "num_timesteps": test_ecog.shape[1],
    }

    logger.info("Evaluation results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info("  %s: %.4f", key, value)
        else:
            logger.info("  %s: %s", key, value)

    return metrics


def decode_phoneme_sequence(
    phoneme_ids: np.ndarray,
    phoneme_map: list = None,
) -> list:
    """
    Convert a sequence of phoneme IDs to human-readable labels.

    Args:
        phoneme_ids: 1D array of phoneme IDs.
        phoneme_map: Mapping from ID to label (uses Config default if None).

    Returns:
        List of phoneme label strings.
    """
    phoneme_map = phoneme_map or Config.PHONEME_MAP

    return [phoneme_map[pid] for pid in phoneme_ids]
