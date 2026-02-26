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

def evolve():
    config_path = Path("neurobridge.config.yaml")
    if not config_path.exists():
        logger.error("Config file not found")
        return

    logger.info("Starting Recursive Evolution Step 1/100")

    # 1. Load Config
    config = NeuroBridgeConfig.from_yaml(config_path)

    # 2. Mutate (Evolve)
    # Perturb learning rate
    original_lr = config.training.learning_rate
    mutation_factor = random.uniform(0.8, 1.2)
    new_lr = original_lr * mutation_factor
    config.training.learning_rate = new_lr

    # Perturb dropout
    original_dropout = config.model.dropout_rate
    config.model.dropout_rate = max(0.1, min(0.5, original_dropout + random.uniform(-0.05, 0.05)))

    logger.info(f"Mutated Hyperparameters: LR {original_lr:.5f} -> {new_lr:.5f}, Dropout {original_dropout:.2f} -> {config.model.dropout_rate:.2f}")

    # 3. Short Training (approx 1/100 of typical 50-100 epochs)
    config.training.max_epochs = 1

    # Ensure directories exist
    config.training.log_dir.mkdir(parents=True, exist_ok=True)
    config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 4. Run Training
    try:
        metrics = train_and_evaluate(config)
        loss = metrics["test_results"].get("loss", float("inf"))
        acc = metrics["test_results"].get("frame_accuracy", 0.0)

        logger.info(f"Evolution Step Complete. Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        print(f"SUCCESS: Evolution step completed. Model evolved with LR={new_lr:.5f}.")

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        # Print stack trace
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    evolve()
