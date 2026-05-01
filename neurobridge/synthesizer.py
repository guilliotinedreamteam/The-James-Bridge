"""
NeuroBridge Speech Synthesizer

Placeholder module for converting decoded phoneme sequences
into audible speech signals. In production, this would integrate
with a vocoder or text-to-speech engine.
"""

import logging
from typing import Optional

import numpy as np

from neurobridge.config import Config

logger = logging.getLogger("neurobridge.synthesizer")


def synthesize_speech_from_phonemes(
    phoneme_id_sequence: np.ndarray,
    sampling_rate: Optional[int] = None,
    phoneme_duration_sec: Optional[float] = None,
    phoneme_map: Optional[list] = None,
) -> np.ndarray:
    """
    Simulate converting a phoneme ID sequence into an audio signal.

    This is a placeholder implementation that generates noise.
    In a real system, this would drive a vocoder (e.g., WaveNet,
    HiFi-GAN) or a TTS engine.

    Args:
        phoneme_id_sequence: 1D array of phoneme IDs.
        sampling_rate: Audio sampling rate (Hz).
        phoneme_duration_sec: Assumed duration of each phoneme (seconds).
        phoneme_map: Mapping from phoneme IDs to labels.

    Returns:
        Simulated audio signal as a 1D float32 array.
    """
    sampling_rate = sampling_rate or Config.AUDIO_SAMPLING_RATE
    phoneme_duration_sec = phoneme_duration_sec or Config.PHONEME_DURATION_SEC
    phoneme_map = phoneme_map or Config.PHONEME_MAP

    # Map IDs to labels
    phoneme_strings = [phoneme_map[pid] for pid in phoneme_id_sequence]

    # Calculate audio dimensions
    total_duration = len(phoneme_id_sequence) * phoneme_duration_sec
    num_samples = int(total_duration * sampling_rate)

    # Generate placeholder audio (small-amplitude noise)
    audio = np.random.randn(num_samples).astype(np.float32) * 0.1

    logger.info(
        "Synthesized speech: %d phonemes → %.2fs at %d Hz (%d samples)",
        len(phoneme_strings),
        total_duration,
        sampling_rate,
        num_samples,
    )
    logger.debug(
        "Phoneme sequence preview: %s",
        " ".join(phoneme_strings[:10]) + ("..." if len(phoneme_strings) > 10 else ""),
    )

    return audio


def phoneme_ids_to_text(
    phoneme_ids: np.ndarray,
    phoneme_map: Optional[list] = None,
) -> str:
    """
    Convert a sequence of phoneme IDs to a space-separated string.

    Args:
        phoneme_ids: 1D array of phoneme IDs.
        phoneme_map: Mapping from ID to label.

    Returns:
        Space-separated string of phoneme labels.
    """
    phoneme_map = phoneme_map or Config.PHONEME_MAP
    return " ".join(phoneme_map[pid] for pid in phoneme_ids)
