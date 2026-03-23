import copy
import random
from pathlib import Path
from typing import Dict, Any

from loguru import logger

from .config import NeuroBridgeConfig
from .training import train_and_evaluate

class Evolver:
    def __init__(self, base_config: NeuroBridgeConfig):
        self.base_config = base_config
        self.best_config = copy.deepcopy(base_config)
        self.best_score = 0.0

    def mutate_config(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        mutated = copy.deepcopy(config)

        # Mutate learning rate
        if random.random() < 0.5:
            mutated.training.learning_rate *= random.uniform(0.5, 2.0)

        # Mutate dropout
        if random.random() < 0.5:
            mutated.model.dropout_rate = min(0.8, max(0.1, mutated.model.dropout_rate + random.uniform(-0.1, 0.1)))

        # Mutate transformer heads
        if random.random() < 0.3:
            mutated.model.num_heads = random.choice([2, 4, 8])

        # Mutate transformer layers
        if random.random() < 0.3:
            mutated.model.transformer_layers = random.choice([1, 2, 3, 4])

        return mutated

    def evolve(self, generations: int, current_generation: int = 1):
        if current_generation > generations:
            logger.info(f"Evolution complete. Best score: {self.best_score:.4f}")
            # Save the best configuration
            output_path = Path("neurobridge.evolved.yaml")
            with open(output_path, "w", encoding="utf-8") as f:
                import yaml
                yaml.dump(self.best_config.to_dict(), f)
            logger.info(f"Saved best config to {output_path}")
            return

        logger.info(f"--- Generation {current_generation}/{generations} ---")

        # In generation 1, evaluate the base config
        if current_generation == 1:
            candidate_config = self.base_config
        else:
            candidate_config = self.mutate_config(self.best_config)

        # Run training
        try:
            metrics = train_and_evaluate(candidate_config)

            # Simple heuristic: we want to maximize test accuracy or something similar
            # Since test_results might be a dict or a list, let's just pick the top3_accuracy
            # from the classification report macro avg f1-score if available, else just a dummy score

            score = 0.0
            if "classification_report" in metrics and "macro avg" in metrics["classification_report"]:
                score = metrics["classification_report"]["macro avg"]["f1-score"]
            elif "test_results" in metrics and isinstance(metrics["test_results"], dict):
                score = metrics["test_results"].get("frame_accuracy", 0.0)

            logger.info(f"Generation {current_generation} score: {score:.4f}")

            if score > self.best_score:
                logger.info(f"New best score found: {score:.4f} > {self.best_score:.4f}")
                self.best_score = score
                self.best_config = copy.deepcopy(candidate_config)

        except Exception as e:
            logger.error(f"Training failed for generation {current_generation}: {e}")

        # Recursive call for the next generation
        self.evolve(generations, current_generation + 1)
