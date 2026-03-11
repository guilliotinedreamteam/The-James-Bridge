import asyncio
from pathlib import Path
from typing import Optional, Dict
from loguru import logger
import copy
import random
import yaml

from .config import NeuroBridgeConfig
from .training import train_and_evaluate

class Evolver:
    def __init__(self, base_config: NeuroBridgeConfig):
        self.base_config = base_config
        self.best_config = copy.deepcopy(base_config)
        self.best_score = 0.0

    def mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        new_config = copy.deepcopy(config)

        # Mutate learning rate
        if random.random() < 0.5:
            new_config.training.learning_rate *= random.uniform(0.5, 2.0)

        # Mutate dropout rate
        if random.random() < 0.5:
            new_config.model.dropout_rate = max(0.0, min(0.9, new_config.model.dropout_rate + random.uniform(-0.1, 0.1)))

        # Mutate units depending on architecture
        if new_config.model.architecture == "transformer":
            if random.random() < 0.5:
                # Need to maintain embed_dim constraints potentially, but num_heads or layers can change
                new_config.model.transformer_layers = max(1, min(8, new_config.model.transformer_layers + random.choice([-1, 0, 1])))
        else: # rnn
            if random.random() < 0.5:
                mutated_units = []
                for unit in new_config.model.rnn_units:
                    mutated_units.append(max(32, int(unit * random.uniform(0.8, 1.2))))
                new_config.model.rnn_units = tuple(mutated_units)

        return new_config

    def evolve(self, generations: int, current_generation: int = 1) -> NeuroBridgeConfig:
        if current_generation > generations:
            logger.info(f"Evolution complete. Best score: {self.best_score}")
            return self.best_config

        logger.info(f"Starting generation {current_generation}/{generations}")

        # Create a mutated config
        candidate_config = self.mutate(self.best_config)

        # Evaluate candidate
        try:
            # We use a very small number of epochs for evolution to be fast, or rely on config
            candidate_config.training.max_epochs = min(3, candidate_config.training.max_epochs)
            metrics = train_and_evaluate(candidate_config)

            # Extract score, e.g., validation accuracy or similar. If not available, use training accuracy
            # Assuming train_and_evaluate returns a dict with 'test_results'
            score = metrics.get('test_results', {}).get('frame_accuracy', 0.0)

            logger.info(f"Generation {current_generation} score: {score}")

            if score > self.best_score:
                logger.info(f"New best score found: {score}")
                self.best_score = score
                self.best_config = candidate_config

                # Save evolved config
                with open("neurobridge.evolved.yaml", "w") as f:
                    yaml.dump(self.best_config.to_dict(), f)

        except Exception as e:
            logger.error(f"Generation {current_generation} failed: {e}")

        # Recursive call for next generation
        return self.evolve(generations, current_generation + 1)
