from __future__ import annotations

import copy
import random
import yaml
import shutil
from pathlib import Path
from loguru import logger
from typing import Optional

from .config import NeuroBridgeConfig
from .training import train_and_evaluate

class Evolver:
    """
    Manages the evolutionary optimization of hyperparameters.
    """
    def __init__(
        self,
        base_config_path: str = "neurobridge.config.yaml",
        output_path: str = "neurobridge.evolved.yaml"
    ):
        self.base_config_path = Path(base_config_path)
        self.output_path = Path(output_path)

        if self.base_config_path.exists():
            self.current_best_config = NeuroBridgeConfig.from_yaml(self.base_config_path)
        else:
            raise FileNotFoundError(f"Base config {base_config_path} not found.")

        # Initial score evaluation or start from -inf
        self.best_score = -float("inf")

    def mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        """
        Applies random mutations to the configuration.
        """
        new_config = copy.deepcopy(config)

        # Mutation Logic
        mutation_prob = 0.5

        # 1. Learning Rate
        if random.random() < mutation_prob:
            factor = random.uniform(0.5, 1.5)
            new_config.training.learning_rate *= factor
            # Clamp
            new_config.training.learning_rate = max(1e-5, min(0.1, new_config.training.learning_rate))

        # 2. Dropout
        if random.random() < mutation_prob:
            delta = random.uniform(-0.1, 0.1)
            new_config.model.dropout_rate += delta
            new_config.model.dropout_rate = max(0.0, min(0.7, new_config.model.dropout_rate))

        # 3. Dense Units
        if random.random() < 0.3:
            # Change by a factor of 2 or keep
            factor = random.choice([0.5, 1.0, 2.0])
            new_config.model.dense_units = int(new_config.model.dense_units * factor)
            new_config.model.dense_units = max(32, new_config.model.dense_units)

        # 4. RNN Units (if applicable)
        if new_config.model.architecture == "rnn" and random.random() < 0.3:
            units_list = list(new_config.model.rnn_units)
            if units_list:
                idx = random.randint(0, len(units_list) - 1)
                factor = random.choice([0.5, 2.0])
                units_list[idx] = int(units_list[idx] * factor)
                units_list[idx] = max(32, units_list[idx])
                new_config.model.rnn_units = tuple(units_list)

        return new_config

    def evaluate(self, config: NeuroBridgeConfig) -> float:
        """
        Runs training for a few epochs and returns the validation accuracy.
        """
        # Use a temporary directory for evolution artifacts to avoid polluting the main artifacts
        temp_dir = Path("runs/evolution_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Modify config for rapid evaluation
        eval_config = copy.deepcopy(config)
        eval_config.training.max_epochs = 1 # One epoch is enough for "one one hundredths of iterations" concept?
                                            # Or just small number for speed.
        eval_config.training.checkpoint_dir = temp_dir / "checkpoints"
        eval_config.training.log_dir = temp_dir / "logs"
        eval_config.training.metrics_path = temp_dir / "metrics.json"

        try:
            metrics = train_and_evaluate(eval_config)
            # Clean up
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            # Get accuracy (prefer validation or test)
            # train_and_evaluate returns test_results
            test_results = metrics.get("test_results", {})
            # Look for keys like 'frame_accuracy', 'accuracy', 'categorical_accuracy'
            score = test_results.get("frame_accuracy", test_results.get("accuracy", 0.0))
            return float(score)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0

    def run(self, generations: int = 10, current_generation: int = 0):
        """
        Main evolution loop.
        """
        if current_generation == 0:
            logger.info(f"Starting evolution for {generations} generations.")

        if current_generation >= generations:
            return

        logger.info(f"=== Generation {current_generation+1} ===")

        candidate = self.mutate(self.current_best_config)

        logger.info(f"Evaluating: LR={candidate.training.learning_rate:.2e}, Drop={candidate.model.dropout_rate:.2f}")
        score = self.evaluate(candidate)

        logger.info(f"Score: {score:.4f} (Current Best: {self.best_score:.4f})")

        if score > self.best_score:
            self.best_score = score
            self.current_best_config = candidate
            logger.success(f"New best config found! Saving to {self.output_path}")
            self.save_best()

        self.run(generations, current_generation + 1)

    def save_best(self):
        # We need to serialize the config properly.
        # NeuroBridgeConfig.to_dict() handles this.
        with open(self.output_path, "w") as f:
            yaml.dump(self.current_best_config.to_dict(), f)

if __name__ == "__main__":
    evolver = Evolver()
    evolver.run(generations=5)
