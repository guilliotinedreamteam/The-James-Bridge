import copy
import random
from pathlib import Path
from loguru import logger
from typing import Dict, Any, List, Optional

from .config import NeuroBridgeConfig
from .training import train_and_evaluate

class Evolver:
    """
    Recursively mutates hyperparameters across generations.
    """
    def __init__(self, base_config: NeuroBridgeConfig):
        self.base_config = base_config
        self.best_config = None
        self.best_score = -1.0
        self.history: List[Dict[str, Any]] = []

    def evolve(self, generations: int, output_path: str = "neurobridge.evolved.yaml"):
        logger.info(f"Starting evolution for {generations} generations")
        self._evolve_recursive(generations, self.base_config)

        if self.best_config:
            self._save_config(self.best_config, Path(output_path))
            logger.info(f"Evolution complete. Best config saved to {output_path} with score {self.best_score:.4f}")
        else:
            logger.warning("Evolution completed but no valid configuration was found.")

    def _evolve_recursive(self, generations_left: int, current_config: NeuroBridgeConfig):
        if generations_left <= 0:
            return

        # 1. Evaluate current config
        try:
            metrics = train_and_evaluate(current_config)

            # Extract score (e.g., from test accuracy)
            score = 0.0
            if "test_results" in metrics and "top3_accuracy" in metrics["test_results"]:
                score = metrics["test_results"]["top3_accuracy"]
            elif "test_results" in metrics and "frame_accuracy" in metrics["test_results"]:
                score = metrics["test_results"]["frame_accuracy"]

            logger.info(f"Generation {generations_left} achieved score: {score:.4f}")

            self.history.append({"generations_left": generations_left, "score": score})

            if score > self.best_score:
                self.best_score = score
                # Create a deep copy using to_dict and from_dict to avoid reference issues
                self.best_config = NeuroBridgeConfig.from_dict(current_config.to_dict())
                logger.info(f"New best score {self.best_score:.4f} found at generation {generations_left}")

        except Exception as e:
            logger.error(f"Evaluation failed at generation {generations_left}: {e}")

        # 2. Mutate to create next config
        next_config = self._mutate(current_config)

        # 3. Recurse
        self._evolve_recursive(generations_left - 1, next_config)

    def _mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        # Create a deep copy for mutation
        mutated_dict = config.to_dict()

        # Mutate learning rate (e.g., scale by 0.5 to 2.0)
        lr_scale = random.uniform(0.5, 2.0)
        new_lr = mutated_dict["training"]["learning_rate"] * lr_scale
        # Clip learning rate
        new_lr = max(1e-5, min(1e-2, new_lr))
        mutated_dict["training"]["learning_rate"] = float(new_lr)

        # Mutate dropout rate
        dropout_shift = random.uniform(-0.1, 0.1)
        new_dropout = mutated_dict["model"].get("dropout_rate", 0.3) + dropout_shift
        new_dropout = max(0.1, min(0.5, new_dropout))
        mutated_dict["model"]["dropout_rate"] = float(new_dropout)

        # Mutate units (e.g., dense_units)
        if random.random() < 0.5:
            current_dense = mutated_dict["model"].get("dense_units", 256)
            choices = [128, 256, 512]
            if current_dense in choices:
                choices.remove(current_dense)
            mutated_dict["model"]["dense_units"] = random.choice(choices)

        # Return new mutated config
        return NeuroBridgeConfig.from_dict(mutated_dict)

    def _save_config(self, config: NeuroBridgeConfig, path: Path):
        import yaml
        config_dict = config.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
