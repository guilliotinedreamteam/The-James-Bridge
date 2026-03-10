import copy
import random
from pathlib import Path
from loguru import logger
import yaml

from .config import NeuroBridgeConfig
from .training import train_and_evaluate


class Evolver:
    def __init__(self, base_config: NeuroBridgeConfig, generations: int = 1):
        self.base_config = base_config
        self.generations = generations
        self.best_config = copy.deepcopy(base_config)
        self.best_score = -1.0

    def mutate(self, config: NeuroBridgeConfig) -> NeuroBridgeConfig:
        mutated = copy.deepcopy(config)

        # Mutate learning rate
        if random.random() < 0.5:
            mutated.training.learning_rate *= random.choice([0.5, 2.0])

        # Mutate dropout
        if random.random() < 0.5:
            mutated.model.dropout_rate = max(0.0, min(0.9, mutated.model.dropout_rate + random.uniform(-0.1, 0.1)))

        # Mutate units depending on architecture
        if mutated.model.architecture == "transformer":
            if random.random() < 0.5:
                mutated.model.num_heads = random.choice([2, 4, 8])
            if random.random() < 0.5:
                mutated.model.transformer_layers = max(1, mutated.model.transformer_layers + random.choice([-1, 1]))
        else:
            if random.random() < 0.5:
                mutated.model.rnn_units = [max(32, u + random.choice([-32, 32])) for u in mutated.model.rnn_units]

        return mutated

    def run(self):
        logger.info(f"Starting evolution for {self.generations} generations")

        current_config = self.base_config

        for gen in range(self.generations):
            logger.info(f"Generation {gen+1}/{self.generations}")
            candidate_config = self.mutate(current_config)

            try:
                metrics = train_and_evaluate(candidate_config)
                # We want to optimize the top3_accuracy or frame_accuracy, if available
                # Let's check test_results from metrics
                test_results = metrics.get("test_results", {})
                score = test_results.get("frame_accuracy", test_results.get("accuracy", 0.0))

                logger.info(f"Generation {gen+1} score: {score}")

                if score > self.best_score:
                    self.best_score = score
                    self.best_config = copy.deepcopy(candidate_config)
                    current_config = candidate_config
                    logger.info(f"New best score: {self.best_score}")
                else:
                    # Occasional acceptance of worse config (simulated annealing like)
                    if random.random() < 0.1:
                        current_config = candidate_config

            except Exception as e:
                logger.error(f"Evaluation failed for generation {gen+1}: {e}")

        # Save best config
        output_path = Path("neurobridge.evolved.yaml")
        with open(output_path, "w") as f:
            yaml.dump(self.best_config.to_dict(), f)

        logger.info(f"Evolution complete. Best config saved to {output_path} with score {self.best_score}")
