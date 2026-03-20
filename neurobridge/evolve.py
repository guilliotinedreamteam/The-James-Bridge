from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from .config import NeuroBridgeConfig
from .training import train_and_evaluate

class Evolver:
    def __init__(self, base_config: NeuroBridgeConfig, generations: int = 100):
        self.base_config = base_config
        self.generations = generations
        self.best_config: Optional[NeuroBridgeConfig] = None
        self.best_score: float = -1.0
        self.out_path = Path("neurobridge.evolved.yaml")

    def mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        new_config = copy.deepcopy(config)
        new_config.training.learning_rate *= random.uniform(0.8, 1.25)
        new_config.model.dropout_rate = min(max(new_config.model.dropout_rate + random.uniform(-0.1, 0.1), 0.0), 0.9)
        new_config.model.dense_units = max(16, int(new_config.model.dense_units * random.uniform(0.8, 1.25)))
        return new_config

    def _evolve_recursive(self, current_config: NeuroBridgeConfig, generation: int) -> None:
        if generation >= self.generations:
            logger.info("Evolution completed.")
            return

        logger.info(f"Generation {generation + 1}/{self.generations}")
        candidate = self.mutate(current_config) if generation > 0 else current_config

        try:
            metrics = train_and_evaluate(candidate)
            score = metrics.get("test_results", {}).get("frame_accuracy", 0.0)
            logger.info(f"Generation {generation + 1} score: {score}")

            if score > self.best_score:
                self.best_score = score
                self.best_config = candidate
                with open(self.out_path, "w", encoding="utf-8") as f:
                    yaml.dump(self.best_config.to_dict(), f)
                logger.info(f"New best score {score}. Saved to {self.out_path}.")
                next_config = candidate
            else:
                next_config = current_config

        except Exception as e:
            logger.error(f"Generation {generation + 1} failed: {e}")
            next_config = current_config

        self._evolve_recursive(next_config, generation + 1)

    def run(self) -> None:
        self._evolve_recursive(self.base_config, 0)
