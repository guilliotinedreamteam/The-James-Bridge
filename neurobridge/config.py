"""
NeuroBridge Configuration

Centralized configuration management for the NeuroBridge system.
All constants, hyperparameters, and paths are defined here and
can be overridden via environment variables.
"""

import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("neurobridge")


class Config:
    """Central configuration for NeuroBridge."""

    # --- Model Architecture ---
    NUM_TIMESTEPS: int = int(os.environ.get("NB_NUM_TIMESTEPS", "100"))
    NUM_FEATURES: int = int(os.environ.get("NB_NUM_FEATURES", "128"))
    NUM_PHONEMES: int = int(os.environ.get("NB_NUM_PHONEMES", "41"))
    LSTM_UNITS: int = int(os.environ.get("NB_LSTM_UNITS", "256"))
    DENSE_UNITS: int = int(os.environ.get("NB_DENSE_UNITS", "128"))

    # --- Training ---
    BATCH_SIZE: int = int(os.environ.get("NB_BATCH_SIZE", "32"))
    EPOCHS: int = int(os.environ.get("NB_EPOCHS", "5"))
    LEARNING_RATE: float = float(os.environ.get("NB_LEARNING_RATE", "0.001"))
    TRAIN_SAMPLES: int = int(os.environ.get("NB_TRAIN_SAMPLES", "1000"))

    # --- Paths ---
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    MODEL_DIR: Path = Path(os.environ.get("NB_MODEL_DIR", str(PROJECT_ROOT / "models")))
    MODEL_FILENAME: str = os.environ.get(
        "NB_MODEL_FILENAME", "neurobridge_decoder.keras"
    )
    REALTIME_MODEL_FILENAME: str = os.environ.get(
        "NB_REALTIME_MODEL_FILENAME", "neurobridge_realtime.keras"
    )

    # --- Data Paths (placeholders for real data) ---
    ECOG_DATA_DIR: str = os.environ.get(
        "NB_ECOG_DATA_DIR", "/path/to/your/real_ecog_data/"
    )
    PHONEME_LABELS_FILE: str = os.environ.get(
        "NB_PHONEME_LABELS_FILE", "/path/to/your/real_phoneme_labels.csv"
    )

    # --- API ---
    API_HOST: str = os.environ.get("NB_API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("NB_API_PORT", "5000"))
    API_DEBUG: bool = os.environ.get("NB_API_DEBUG", "false").lower() == "true"

    # --- Audio ---
    AUDIO_SAMPLING_RATE: int = int(os.environ.get("NB_AUDIO_SAMPLE_RATE", "16000"))
    PHONEME_DURATION_SEC: float = float(os.environ.get("NB_PHONEME_DURATION", "0.1"))

    # --- Phoneme Map ---
    # Conceptual mapping from phoneme ID to label.
    # Replace with actual phoneme dictionary for real data.
    PHONEME_MAP: list = [f"PH_{i}" for i in range(NUM_PHONEMES)]

    @classmethod
    def model_save_path(cls) -> Path:
        """Full path for saving/loading the offline decoder model."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODEL_DIR / cls.MODEL_FILENAME

    @classmethod
    def realtime_model_save_path(cls) -> Path:
        """Full path for saving/loading the real-time decoder model."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODEL_DIR / cls.REALTIME_MODEL_FILENAME

    @classmethod
    def summary(cls) -> str:
        """Return a human-readable summary of the current configuration."""
        lines = [
            "NeuroBridge Configuration",
            "=" * 40,
            f"  Timesteps:      {cls.NUM_TIMESTEPS}",
            f"  Features:       {cls.NUM_FEATURES}",
            f"  Phonemes:       {cls.NUM_PHONEMES}",
            f"  LSTM units:     {cls.LSTM_UNITS}",
            f"  Batch size:     {cls.BATCH_SIZE}",
            f"  Epochs:         {cls.EPOCHS}",
            f"  Learning rate:  {cls.LEARNING_RATE}",
            f"  Model dir:      {cls.MODEL_DIR}",
            f"  API port:       {cls.API_PORT}",
        ]
        return "\n".join(lines)
