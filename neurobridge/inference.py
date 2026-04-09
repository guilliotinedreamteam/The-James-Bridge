"""
NeuroBridge Real-time Inference

Provides classes and functions for real-time ECoG-to-Phoneme
decoding from single neural signal frames.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from neurobridge.config import Config

logger = logging.getLogger("neurobridge.inference")


class RealtimeDecoder:
    """
    Real-time phoneme decoder for single-frame ECoG inference.

    Manages model loading and provides a simple API for
    predicting phonemes from individual ECoG frames.

    Usage:
        decoder = RealtimeDecoder()
        decoder.load()  # or decoder.load("/path/to/model.keras")
        probs = decoder.predict(ecog_frame)  # shape (features,)
        phoneme_id = decoder.predict_label(ecog_frame)
    """

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self._features = Config.NUM_FEATURES
        self._num_phonemes = Config.NUM_PHONEMES

    def load(self, model_path: Optional[str] = None) -> None:
        """Load a trained real-time model from disk."""
        path = model_path or str(Config.realtime_model_save_path())

        if not Path(path).exists():
            # Fall back to offline model
            fallback = str(Config.model_save_path())
            if Path(fallback).exists():
                logger.warning(
                    "Real-time model not found at %s. "
                    "Falling back to offline model at %s",
                    path, fallback,
                )
                path = fallback
            else:
                raise FileNotFoundError(
                    f"No model found at {path} or {fallback}. "
                    "Train a model first."
                )

        logger.info("Loading real-time model from: %s", path)
        self.model = tf.keras.models.load_model(path)
        logger.info("Real-time model loaded successfully.")

    def predict(self, ecog_frame: np.ndarray) -> np.ndarray:
        """
        Predict phoneme probabilities from a single ECoG frame.

        Args:
            ecog_frame: 1D array of shape (features,).

        Returns:
            1D array of phoneme probabilities, shape (num_phonemes,).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Validate input
        if ecog_frame.ndim != 1 or ecog_frame.shape[0] != self._features:
            raise ValueError(
                f"Expected 1D array of shape ({self._features},), "
                f"got shape {ecog_frame.shape}"
            )

        # Reshape to (1, 1, features) for single-frame prediction
        frame = ecog_frame.reshape(1, 1, -1).astype(np.float32)

        # PHASE 9: Latency Optimization.
        # model.predict() has massive overhead. Direct invocation is vastly faster for real-time.
        probs_tensor = self.model(frame, training=False)

        # Flatten: (1, num_phonemes) → (num_phonemes,)
        return probs_tensor.numpy().flatten()

    def predict_label(
        self,
        ecog_frame: np.ndarray,
        phoneme_map: list = None,
    ) -> str:
        """
        Predict the most likely phoneme label from a single frame.

        Args:
            ecog_frame: 1D array of shape (features,).
            phoneme_map: List mapping phoneme IDs to labels.

        Returns:
            The predicted phoneme label string.
        """
        phoneme_map = phoneme_map or Config.PHONEME_MAP
        probs = self.predict(ecog_frame)
        best_id = int(np.argmax(probs))
        return phoneme_map[best_id]

    def predict_top_k(
        self,
        ecog_frame: np.ndarray,
        k: int = 5,
        phoneme_map: list = None,
    ) -> list:
        """
        Get the top-k predicted phonemes with probabilities.

        Args:
            ecog_frame: 1D array of shape (features,).
            k: Number of top predictions to return.
            phoneme_map: List mapping phoneme IDs to labels.

        Returns:
            List of (label, probability) tuples, sorted by probability desc.
        """
        phoneme_map = phoneme_map or Config.PHONEME_MAP
        probs = self.predict(ecog_frame)
        top_indices = np.argsort(probs)[::-1][:k]
        return [
            (phoneme_map[i], float(probs[i]))
            for i in top_indices
        ]


def predict_realtime_phoneme(
    ecog_frame: np.ndarray,
    model: tf.keras.Model,
) -> np.ndarray:
    """
    Standalone function for real-time phoneme prediction.

    Args:
        ecog_frame: 1D array of shape (features,).
        model: A loaded Keras model.

    Returns:
        1D array of phoneme probabilities.
    """
    frame = ecog_frame.reshape(1, 1, -1).astype(np.float32)

    # PHASE 9: Latency Optimization.
    # model.predict() has massive overhead. Direct invocation is vastly faster for real-time.
    probs_tensor = model(frame, training=False)

    return probs_tensor.numpy().flatten()
