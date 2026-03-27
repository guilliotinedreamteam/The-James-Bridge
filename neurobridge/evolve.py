import json
import random
import yaml
from pathlib import Path
from loguru import logger
from copy import deepcopy

from .config import NeuroBridgeConfig
from .training import train_and_evaluate

class Evolver:
    def __init__(self, base_config: NeuroBridgeConfig, generations: int = 100):
        self.base_config = base_config
        self.generations = generations
        self.best_config = None
        self.best_accuracy = -1.0
        self.current_generation = 0

    def mutate_config(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        new_config = deepcopy(config)
        # Mutate learning rate
        if random.random() < 0.5:
            new_config.training.learning_rate *= random.choice([0.5, 0.8, 1.2, 2.0])
        # Mutate dropout rate
        if random.random() < 0.5:
            new_config.model.dropout_rate = min(max(new_config.model.dropout_rate + random.uniform(-0.1, 0.1), 0.1), 0.5)
        # Mutate rnn units (or similar architectural param)
        if random.random() < 0.5:
            if hasattr(new_config.model, 'rnn_units'):
                 new_units = [max(64, u + random.choice([-64, 0, 64])) for u in new_config.model.rnn_units]
                 new_config.model.rnn_units = new_units
            elif hasattr(new_config.model, 'transformer_layers'):
                 new_config.model.transformer_layers = max(1, new_config.model.transformer_layers + random.choice([-1, 0, 1]))
        return new_config

    def run(self):
        logger.info(f"Starting evolution over {self.generations} generations...")
        self.evolve_recursive(self.base_config, self.generations)
        if self.best_config:
            self.save_best_config()
        logger.info(f"Evolution complete. Best accuracy: {self.best_accuracy}")

    def evolve_recursive(self, current_config: NeuroBridgeConfig, remaining_generations: int):
        if remaining_generations <= 0:
            return

        self.current_generation += 1
        logger.info(f"--- Generation {self.current_generation} ---")

        mutated_config = self.mutate_config(current_config) if self.current_generation > 1 else current_config

        try:
            metrics = train_and_evaluate(mutated_config)
            test_results = metrics.get('test_results', {})
            # Depending on metric name, 'frame_accuracy' or similar
            accuracy = test_results.get('frame_accuracy', 0.0)

            logger.info(f"Generation {self.current_generation} accuracy: {accuracy}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_config = deepcopy(mutated_config)
                logger.info(f"New best model found! Accuracy: {self.best_accuracy}")

            # If accuracy is very low, maybe don't use it as the base for the next generation
            # For simplicity, we'll always use the best found so far or the newly mutated one
            # Let's use the newly mutated one if it's decent, or fallback to best
            next_base = mutated_config if accuracy > (self.best_accuracy * 0.9) else self.best_config

        except Exception as e:
            logger.error(f"Generation {self.current_generation} failed: {e}")
            next_base = current_config # fallback

        self.evolve_recursive(next_base, remaining_generations - 1)

    def save_best_config(self, filepath="neurobridge.evolved.yaml"):
        if self.best_config:
            with open(filepath, 'w') as f:
                yaml.dump(self.best_config.to_dict(), f)
            logger.info(f"Saved best config to {filepath}")
