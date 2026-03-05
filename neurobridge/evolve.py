import random
import yaml
from pathlib import Path
from loguru import logger
import tensorflow as tf

# Ensure paths are correct relative to execution
import sys
import os
sys.path.append(os.getcwd())

from neurobridge.config import NeuroBridgeConfig
from neurobridge.training import train_and_evaluate

class Evolver:
    def __init__(self, base_config_path: str = "neurobridge.config.yaml", output_path: str = "neurobridge.evolved.yaml"):
        self.base_config_path = Path(base_config_path)
        self.output_path = Path(output_path)
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config {base_config_path} not found.")
        self.best_config = NeuroBridgeConfig.from_yaml(self.base_config_path)
        self.best_score = -float('inf')

    def mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        import copy
        new_config = copy.deepcopy(config)
        mutation_prob = 0.5
        if random.random() < mutation_prob:
            factor = random.uniform(0.5, 1.5)
            new_config.training.learning_rate = max(1e-5, min(0.1, new_config.training.learning_rate * factor))
        if random.random() < mutation_prob:
            delta = random.uniform(-0.1, 0.1)
            new_config.model.dropout_rate = max(0.0, min(0.7, new_config.model.dropout_rate + delta))
        if random.random() < 0.3:
            factor = random.choice([0.5, 1.0, 2.0])
            new_config.model.dense_units = max(32, int(new_config.model.dense_units * factor))
        return new_config

    def evaluate(self, config: NeuroBridgeConfig) -> float:
        import copy
        eval_config = copy.deepcopy(config)
        eval_config.training.max_epochs = 1
        try:
            metrics = train_and_evaluate(eval_config)
            test_results = metrics.get("test_results", {})
            return float(test_results.get("frame_accuracy", test_results.get("accuracy", 0.0)))
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0

    def run(self, generations: int = 100):
        logger.info(f"Starting Recursive Evolution for {generations} iterations")
        for g in range(generations):
            logger.info(f"=== Generation {g+1}/{generations} ===")
            candidate = self.mutate(self.best_config)
            logger.info(f"Evaluating: LR={candidate.training.learning_rate:.2e}, Drop={candidate.model.dropout_rate:.2f}")
            score = self.evaluate(candidate)
            logger.info(f"Score: {score:.4f} (Current Best: {self.best_score:.4f})")
            if score > self.best_score:
                self.best_score = score
                self.best_config = candidate
                logger.success(f"New best config found! Saving to {self.output_path}")

                # Custom recursive serialization for paths in NeuroBridgeConfig.to_dict
                def _serialize_paths(d):
                    for k, v in d.items():
                        if isinstance(v, Path):
                            d[k] = str(v)
                        elif isinstance(v, dict):
                            d[k] = _serialize_paths(v)
                    return d

                config_dict = _serialize_paths(self.best_config.to_dict())
                with open(self.output_path, "w") as f:
                    yaml.dump(config_dict, f)

if __name__ == "__main__":
    evolver = Evolver()
    evolver.run(generations=100)
