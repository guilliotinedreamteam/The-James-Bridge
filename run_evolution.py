#!/usr/bin/env python3
import argparse
import sys
import logging
import numpy as np

from neurobridge.data.ingestion import ECoGIngestionPipeline
from neurobridge.processing.signal import SignalProcessor
from neurobridge.evolve import Evolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Neurobridge Recursive Hyperparameter Evolution")
    parser.add_argument("--file", type=str, default="sample_data/clinical_validation_sample.edf", help="Path to medical dataset")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to evolve")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.file}...")
    pipeline = ECoGIngestionPipeline(expected_channels=1)
    raw_data = pipeline.load_medical_dataset(args.file)
    data_array = pipeline.extract_numpy_arrays(raw_data)

    processor = SignalProcessor()
    downsampled = processor.downsample_signals(data_array, current_freq=int(raw_data.info['sfreq']))
    normalized = processor.z_score_normalize(downsampled)
    shaped = processor.shape_sequences(normalized)

    batch_size, timesteps, channels = shaped.shape

    # Generate synthetic labels for evolution demonstration
    # Real use case would use alignment logic as in CLI
    logger.info("Generating synthetic labels for evolution...")
    random_phonemes = np.random.randint(0, 41, size=(batch_size, timesteps))
    y_train = np.eye(41)[random_phonemes]

    evolver = Evolver(x_data=shaped, y_data=y_train, epochs_per_gen=1)
    evolver.run(generations=args.generations)

if __name__ == "__main__":
    main()
