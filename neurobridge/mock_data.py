from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import savemat
from loguru import logger

from .config import NeuroBridgeConfig

def generate_mock_data(config: NeuroBridgeConfig, num_files: int = 10, duration_sec: float = 5.0):
    """
    Generates synthetic ECoG data and phoneme labels based on the configuration.
    """
    ecog_dir = config.dataset.ecog_dir
    labels_dir = config.dataset.labels_dir

    ecog_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    fs = config.dataset.sampling_rate_hz
    num_channels = config.dataset.num_features
    total_samples = int(duration_sec * fs)

    phonemes = config.dataset.phonemes

    logger.info(f"Generating {num_files} mock datasets in {ecog_dir} and {labels_dir}")

    for i in range(num_files):
        file_id = f"mock_{i:03d}"

        # 1. Generate ECoG Signal (Random Noise + Sine waves)
        # Using a mix of frequencies to simulate "bands"
        t = np.linspace(0, duration_sec, total_samples)
        signal = np.random.randn(total_samples, num_channels) * 0.5

        # Add some structure
        for ch in range(num_channels):
            freq = np.random.uniform(1, 150)
            phase = np.random.uniform(0, 2*np.pi)
            signal[:, ch] += np.sin(2 * np.pi * freq * t + phase)

        # Save ECoG
        if config.dataset.file_extension == ".mat":
            mat_path = ecog_dir / f"{file_id}.mat"
            key = config.dataset.mat_key or "ecog_data"
            savemat(mat_path, {key: signal})
        elif config.dataset.file_extension == ".npy":
            npy_path = ecog_dir / f"{file_id}.npy"
            np.save(npy_path, signal)
        else:
            logger.warning(f"Unsupported extension {config.dataset.file_extension} for mock generation.")

        # 2. Generate Phoneme Labels
        # Create random intervals
        intervals = []
        current_time = 0.0
        while current_time < duration_sec:
            ph = np.random.choice(phonemes)
            dur = np.random.uniform(0.05, 0.2) # 50ms to 200ms
            end_time = min(current_time + dur, duration_sec)

            intervals.append({
                config.dataset.label_columns["start"]: current_time,
                config.dataset.label_columns["end"]: end_time,
                config.dataset.label_columns["label"]: ph
            })

            current_time = end_time
            if current_time >= duration_sec:
                break

        df = pd.DataFrame(intervals)
        csv_path = labels_dir / f"{file_id}{config.dataset.label_extension}"
        df.to_csv(csv_path, index=False)

    logger.info("Mock data generation complete.")

if __name__ == "__main__":
    # Allow running directly to generate data
    from .config import NeuroBridgeConfig
    # Assume default config file exists or create a default object
    try:
        cfg = NeuroBridgeConfig.from_yaml("neurobridge.config.yaml")
    except FileNotFoundError:
        logger.warning("Config file not found, creating default config.")
        # Create minimal valid config for testing
        from .config import DatasetConfig, ModelConfig, TrainingConfig, RealtimeConfig, SpeechConfig
        cfg = NeuroBridgeConfig(
            dataset=DatasetConfig(ecog_dir="data/ecog", labels_dir="data/labels"),
            model=ModelConfig(),
            training=TrainingConfig(),
            realtime=RealtimeConfig(),
            speech=SpeechConfig()
        )

    generate_mock_data(cfg)
