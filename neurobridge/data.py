"""
NeuroBridge Data Loading & Preprocessing

Provides functions for generating mock data (testing/development)
and loading real ECoG/phoneme data (production).
"""

import logging
from typing import Generator, Optional, Tuple

import numpy as np

from neurobridge.config import Config

logger = logging.getLogger("neurobridge.data")


# ---------------------------------------------------------------------------
# Mock data generators (for development & testing)
# ---------------------------------------------------------------------------

def generate_mock_ecog_data(
    num_samples: int,
    timesteps: int = None,
    features: int = None,
) -> np.ndarray:
    """
    Generate random mock ECoG data for testing.

    Args:
        num_samples: Number of samples to generate.
        timesteps: Sequence length per sample.
        features: Number of ECoG channels.

    Returns:
        Array of shape (num_samples, timesteps, features), dtype float32.
    """
    timesteps = timesteps or Config.NUM_TIMESTEPS
    features = features or Config.NUM_FEATURES

    data = np.random.rand(num_samples, timesteps, features).astype(np.float32)
    logger.debug(
        "Generated mock ECoG data: shape=%s", data.shape,
    )
    return data


def generate_mock_phoneme_labels(
    num_samples: int,
    timesteps: int = None,
    num_classes: int = None,
) -> np.ndarray:
    """
    Generate random mock one-hot phoneme labels for testing.

    Args:
        num_samples: Number of samples.
        timesteps: Sequence length per sample.
        num_classes: Number of phoneme classes.

    Returns:
        One-hot array of shape (num_samples, timesteps, num_classes), float32.
    """
    import tensorflow as tf

    timesteps = timesteps or Config.NUM_TIMESTEPS
    num_classes = num_classes or Config.NUM_PHONEMES

    sparse = np.random.randint(0, num_classes, size=(num_samples, timesteps))
    onehot = tf.keras.utils.to_categorical(sparse, num_classes=num_classes)
    labels = onehot.astype(np.float32)

    logger.debug("Generated mock phoneme labels: shape=%s", labels.shape)
    return labels


def create_data_generator(
    batch_size: int = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Infinite generator that yields batches of mock ECoG data and labels.

    Args:
        batch_size: Number of samples per batch.

    Yields:
        Tuple of (ecog_batch, phoneme_labels_batch).
    """
    batch_size = batch_size or Config.BATCH_SIZE

    while True:
        ecog = generate_mock_ecog_data(batch_size)
        labels = generate_mock_phoneme_labels(batch_size)
        yield ecog, labels


# ---------------------------------------------------------------------------
# Real data loaders (placeholders — implement for your dataset)
# ---------------------------------------------------------------------------

def load_real_ecog_data(
    data_path: str,
    timesteps: int = None,
    features: int = None,
    sampling_rate: int = 1000,
    target_downsample_rate: int = 100,
) -> Optional[np.ndarray]:
    """
    Load and preprocess real ECoG data from a .mat file.

    This is a placeholder implementation. Replace the body of this function
    with your actual data loading and preprocessing pipeline.

    Args:
        data_path: Path to the ECoG data file (e.g., .mat).
        timesteps: Target number of timesteps per sample.
        features: Number of ECoG features (channels).
        sampling_rate: Original sampling rate (Hz).
        target_downsample_rate: Target sampling rate after downsampling (Hz).

    Returns:
        Preprocessed ECoG data of shape (timesteps, features), or None on error.
    """
    from scipy.io import loadmat

    timesteps = timesteps or Config.NUM_TIMESTEPS
    features = features or Config.NUM_FEATURES

    logger.info("Loading ECoG data from: %s", data_path)

    try:
        mat_data = loadmat(data_path)
        raw = mat_data.get("ecog_data")
        if raw is None:
            raise ValueError(
                f"'ecog_data' key not found in {data_path}. "
                "Check the .mat file structure."
            )

        raw = raw.astype(np.float32)

        # Downsample if needed
        if sampling_rate > target_downsample_rate:
            factor = sampling_rate // target_downsample_rate
            if factor > 1:
                raw = raw[::factor, :]

        # Trim or pad to match target timesteps
        if raw.shape[0] > timesteps:
            raw = raw[:timesteps, :]
        elif raw.shape[0] < timesteps:
            padding = np.zeros(
                (timesteps - raw.shape[0], features), dtype=np.float32,
            )
            raw = np.vstack([raw, padding])

        # Validate feature dimension
        if raw.shape[1] != features:
            raise ValueError(
                f"Feature mismatch: expected {features}, got {raw.shape[1]}",
            )

        logger.info("Preprocessed ECoG data shape: %s", raw.shape)
        return raw

    except FileNotFoundError:
        logger.error("ECoG data file not found: %s", data_path)
        return None
    except Exception as e:
        logger.error("Error processing ECoG data from %s: %s", data_path, e)
        return None


def load_real_phoneme_labels(
    labels_path: str,
    timesteps: int = None,
    num_classes: int = None,
) -> Optional[np.ndarray]:
    """
    Load and preprocess real phoneme labels.

    This is a placeholder implementation. Replace the body with your
    actual label loading and alignment pipeline (e.g., CTC preparation,
    dynamic time warping).

    Args:
        labels_path: Path to the phoneme labels file (e.g., CSV).
        timesteps: Target number of timesteps for alignment.
        num_classes: Total number of unique phonemes.

    Returns:
        One-hot encoded labels of shape (timesteps, num_classes), or None.
    """
    import tensorflow as tf

    timesteps = timesteps or Config.NUM_TIMESTEPS
    num_classes = num_classes or Config.NUM_PHONEMES

    logger.info("Loading phoneme labels from: %s", labels_path)

    try:
        # PLACEHOLDER: Replace with actual label loading logic.
        # For now, generates random labels to simulate successful loading.
        sparse = np.random.randint(0, num_classes, size=(timesteps,))
        onehot = tf.keras.utils.to_categorical(
            sparse, num_classes=num_classes,
        ).astype(np.float32)

        logger.info("Preprocessed phoneme labels shape: %s", onehot.shape)
        return onehot

    except FileNotFoundError:
        logger.error("Phoneme labels file not found: %s", labels_path)
        return None
    except Exception as e:
        logger.error("Error processing phoneme labels from %s: %s", labels_path, e)
        return None
