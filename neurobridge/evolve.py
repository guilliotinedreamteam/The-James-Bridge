import asyncio
from pathlib import Path
import random
import yaml
from loguru import logger
from typing import Optional

from .config import NeuroBridgeConfig
from .training import train_and_evaluate

class Evolver:
    def __init__(self, config: NeuroBridgeConfig):
        self.config = config
        self.best_config = None
        self.best_score = float('-inf')

    def mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        # Create a deep copy via dict serialization
        new_config_dict = config.to_dict()
        new_config = NeuroBridgeConfig.from_dict(new_config_dict)

        # Mutate learning rate
        lr_factor = random.uniform(0.5, 2.0)
        new_config.training.learning_rate *= lr_factor

        # Mutate dropout
        dropout_shift = random.uniform(-0.1, 0.1)
        new_config.model.dropout_rate = max(0.0, min(0.9, new_config.model.dropout_rate + dropout_shift))

        # Mutate units depending on architecture
        if new_config.model.architecture == "transformer":
            # Mutate ff_dim
            if random.random() < 0.5:
                shift = random.choice([-128, -64, 64, 128])
                new_config.model.ff_dim = max(64, new_config.model.ff_dim + shift)
            # Mutate transformer layers
            if random.random() < 0.3:
                shift = random.choice([-1, 1])
                new_config.model.transformer_layers = max(1, new_config.model.transformer_layers + shift)
        else:
            # Mutate rnn_units
            if random.random() < 0.5:
                new_units = list(new_config.model.rnn_units)
                idx = random.randint(0, len(new_units) - 1)
                shift = random.choice([-64, 64])
                new_units[idx] = max(32, new_units[idx] + shift)
                new_config.model.rnn_units = tuple(new_units)

        return new_config

    def evolve(self, generations: int) -> NeuroBridgeConfig:
        for current_gen in range(1, generations + 1):
            logger.info(f"--- Generation {current_gen}/{generations} ---")

            # Mutate from best or current
            candidate_config = self.mutate(self.best_config if self.best_config else self.config)

            # We only train for a few epochs per generation to save time
            candidate_config.training.max_epochs = 3

            try:
                metrics = train_and_evaluate(candidate_config)
                # Use top3_accuracy if available, else frame_accuracy
                test_results = metrics.get("test_results", {})
                score = test_results.get("top3_accuracy", test_results.get("frame_accuracy", 0.0))

                logger.info(f"Generation {current_gen} scored {score:.4f}")

                if score > self.best_score:
                    logger.info(f"New best score! {score:.4f} > {self.best_score:.4f}")
                    self.best_score = score
                    self.best_config = candidate_config

            except Exception as e:
                logger.error(f"Generation {current_gen} failed: {e}")

        logger.info("Evolution complete.")
        if self.best_config:
            best_path = Path("neurobridge.evolved.yaml")
            with open(best_path, "w", encoding="utf-8") as f:
                yaml.dump(self.best_config.to_dict(), f)
            logger.info(f"Saved best config to {best_path}")
            return self.best_config
        return self.config
