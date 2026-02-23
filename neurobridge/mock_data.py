import numpy as np
import pandas as pd
from scipy.io import savemat
from pathlib import Path
import os
import random
import sys

# Add current directory to path to import neurobridge
sys.path.append(os.getcwd())
from neurobridge.config import _default_phonemes

def generate_mock_data():
    # Configuration
    NUM_SAMPLES = 20
    NUM_TIMESTEPS = 2000 # 2s duration
    NUM_FEATURES = 128
    SAMPLING_RATE = 1000

    ECOG_DIR = Path("data/ecog")
    LABELS_DIR = Path("data/labels")

    ECOG_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Use full default phoneme set to match config
    phonemes = _default_phonemes()
    # Remove 'sil' if it's already there (it is), we'll use it randomly

    print(f"Generating {NUM_SAMPLES} mock samples in {ECOG_DIR} and {LABELS_DIR} using {len(phonemes)} phonemes...")

    for i in range(NUM_SAMPLES):
        # Generate ECoG data: Random noise + sine wave
        time = np.linspace(0, NUM_TIMESTEPS/SAMPLING_RATE, NUM_TIMESTEPS)
        signal = np.random.randn(NUM_TIMESTEPS, NUM_FEATURES).astype(np.float32) * 0.5

        # Add some structure (sine waves) to make it look slightly real
        for ch in range(5): # Add signal to first 5 channels
            freq = random.uniform(10, 100)
            signal[:, ch] += np.sin(2 * np.pi * freq * time).astype(np.float32)

        filename = f"sample_{i:03d}"
        mat_path = ECOG_DIR / f"{filename}.mat"
        # Save as 'ecog_data' key as expected by default config
        savemat(mat_path, {"ecog_data": signal})

        # Generate Labels: CSV with start_sec, end_sec, phoneme
        intervals = []
        current_time = 0.0
        total_duration = NUM_TIMESTEPS / SAMPLING_RATE

        while current_time < total_duration:
            # Phoneme duration between 50ms and 150ms (faster speech)
            duration = random.uniform(0.05, 0.15)
            end_time = min(current_time + duration, total_duration)

            ph = random.choice(phonemes)
            intervals.append({
                "start_sec": float(current_time),
                "end_sec": float(end_time),
                "phoneme": ph
            })
            current_time = end_time

        df = pd.DataFrame(intervals)
        csv_path = LABELS_DIR / f"{filename}.csv"
        df.to_csv(csv_path, index=False)

    print("Mock data generation complete.")

if __name__ == "__main__":
    generate_mock_data()
