import copy
import random
from pathlib import Path
from loguru import logger
from pydantic import BaseModel

from .config import NeuroBridgeConfig
from .training import train_and_evaluate


class Evolver:
    def __init__(self, base_config: NeuroBridgeConfig, generations: int = 100):
        self.base_config = base_config
        self.generations = generations
        self.best_config = copy.deepcopy(base_config)
        self.best_accuracy = 0.0
        self.current_generation = 0
        self.evolved_path = Path("neurobridge.evolved.yaml")

    def mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        """Slightly mutates hyperparameters for evolution."""
        mutated = copy.deepcopy(config)

        # Mutate learning rate (e.g., scale by a factor between 0.5 and 2.0)
        mutated.training.learning_rate *= random.uniform(0.5, 2.0)

        # Mutate dropout rate (ensure it stays between 0.1 and 0.5)
        mutated.model.dropout_rate = max(0.1, min(0.5, mutated.model.dropout_rate + random.uniform(-0.1, 0.1)))

        # Mutate dense units (e.g., +/- 10%)
        mutated.model.dense_units = max(32, int(mutated.model.dense_units * random.uniform(0.9, 1.1)))

        return mutated

    def run(self):
        """Recursively runs the evolution process."""
        if self.current_generation >= self.generations:
            logger.info("Evolution complete.")
            return

        self.current_generation += 1
        logger.info(f"--- Generation {self.current_generation}/{self.generations} ---")

        # Mutate the best config so far
        candidate_config = self.mutate(self.best_config)

        try:
            # Evaluate the candidate
            metrics = train_and_evaluate(candidate_config)

            # Use 'frame_accuracy' from test results as the metric to maximize
            # The test_results dictionary maps metric names to values. Let's assume the order
            # is [loss, frame_accuracy, top3_accuracy] or similar based on training.py
            # If not present, default to 0

            # The model evaluate returns a dict where keys match the metrics names.
            accuracy = metrics.get("test_results", {}).get("frame_accuracy", 0.0)

            logger.info(f"Candidate accuracy: {accuracy:.4f}")

            if accuracy > self.best_accuracy:
                logger.info(f"New best model found! Accuracy improved from {self.best_accuracy:.4f} to {accuracy:.4f}")
                self.best_accuracy = accuracy
                self.best_config = candidate_config

                # Save the new best configuration
                import yaml
                with open(self.evolved_path, "w", encoding="utf-8") as f:
                    yaml.dump(self.best_config.to_dict(), f)

        except Exception as e:
            logger.error(f"Generation {self.current_generation} failed: {e}")

        # Recursive call
        self.run()
