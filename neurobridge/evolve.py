from loguru import logger
import copy
import random
import yaml
from pathlib import Path
from .config import NeuroBridgeConfig
from .training import train_and_evaluate

class Evolver:
    def __init__(self, base_config: NeuroBridgeConfig):
        self.base_config = base_config
        self.best_config = copy.deepcopy(base_config)
        self.best_score = -1.0

    def mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        new_config = copy.deepcopy(config)
        new_config.training.learning_rate *= random.uniform(0.5, 2.0)
        new_config.model.dropout_rate = max(0.0, min(0.9, new_config.model.dropout_rate + random.uniform(-0.1, 0.1)))

        if new_config.model.architecture == "rnn":
            units = list(new_config.model.rnn_units)
            if units:
                idx = random.randint(0, len(units) - 1)
                units[idx] = max(32, units[idx] + random.choice([-32, 32]))
            new_config.model.rnn_units = tuple(units)
        elif new_config.model.architecture == "transformer":
            new_config.model.ff_dim = max(64, new_config.model.ff_dim + random.choice([-64, 64]))

        return new_config

    def _evolve_recursive(self, current_config: NeuroBridgeConfig, current_gen: int, max_gens: int):
        if current_gen >= max_gens:
            return self.best_config

        logger.info(f"Generation {current_gen+1}/{max_gens}")
        candidate_config = self.mutate(current_config) if current_gen > 0 else current_config

        try:
            metrics = train_and_evaluate(candidate_config)
            score = metrics.get('test_results', {}).get('frame_accuracy', 0.0)
            logger.info(f"Generation {current_gen+1} score: {score}")

            if score > self.best_score:
                self.best_score = score
                self.best_config = copy.deepcopy(candidate_config)
                self.save_best_config(Path("neurobridge.evolved.yaml"))
                return self._evolve_recursive(candidate_config, current_gen + 1, max_gens)
            else:
                return self._evolve_recursive(current_config, current_gen + 1, max_gens)
        except Exception as e:
            logger.error(f"Generation {current_gen+1} failed: {e}")
            return self._evolve_recursive(current_config, current_gen + 1, max_gens)

    def run(self, generations: int):
        logger.info(f"Starting recursive evolution for {generations} generations")
        return self._evolve_recursive(self.base_config, 0, generations)

    def save_best_config(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.best_config.to_dict(), f)
        logger.info(f"Saved best config to {path} with score {self.best_score}")
