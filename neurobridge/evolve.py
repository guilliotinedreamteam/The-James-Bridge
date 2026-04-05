import dataclasses
import pathlib
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class NeuroBridgeConfig:
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    lstm_units: int = 256
    dense_units: int = 128

    def to_dict(self):
        def _serialize(obj):
            if isinstance(obj, pathlib.Path):
                return str(obj)
            if isinstance(obj, tuple):
                return [_serialize(item) for item in obj]
            if isinstance(obj, list):
                return [_serialize(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            return obj

        return _serialize(dataclasses.asdict(self))

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f)


class Evolver:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.best_config = None
        self.best_score = float('-inf')
        self.evaluations = 0

    def _evolve_recursive(self, current_config: NeuroBridgeConfig, generations_left: int, max_generations: int):
        from neurobridge.model.decoder import NeurobridgeDecoder
        from neurobridge.training.trainer import ModelTrainer
        import random

        if generations_left <= 0:
            return

        self.evaluations += 1
        logger.info(f"Generation {max_generations - generations_left + 1}/{max_generations} - Config: {current_config}")

        timesteps = self.x_data.shape[1]
        channels = self.x_data.shape[2]

        decoder = NeurobridgeDecoder(
            timesteps=timesteps,
            channels=channels,
            lstm_units=current_config.lstm_units,
            dropout_rate=current_config.dropout_rate,
            dense_units=current_config.dense_units
        )
        model = decoder.build_offline_decoder()

        # Need to compile with dynamic learning rate
        import tensorflow as tf
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=current_config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        trainer = ModelTrainer(model=model, checkpoint_dir="evolution_checkpoints")
        history = trainer.train(
            x_train=self.x_data,
            y_train=self.y_data,
            epochs=2, # Small epochs for quick evolution
            batch_size=8,
            model_name=f"evolve_gen_{self.evaluations}"
        )

        score = history.history['accuracy'][-1]

        if score > self.best_score:
            self.best_score = score
            self.best_config = dataclasses.replace(current_config)
            self.best_config.save("neurobridge.evolved.yaml")
            logger.info(f"New best score: {score:.4f} with config {self.best_config}")

        # Mutate
        next_config = dataclasses.replace(current_config)
        mutation_choice = random.choice(["learning_rate", "dropout_rate", "lstm_units", "dense_units"])

        if mutation_choice == "learning_rate":
            next_config.learning_rate *= random.choice([0.5, 0.8, 1.2, 2.0])
        elif mutation_choice == "dropout_rate":
            next_config.dropout_rate = max(0.1, min(0.6, next_config.dropout_rate + random.uniform(-0.1, 0.1)))
        elif mutation_choice == "lstm_units":
            next_config.lstm_units = int(next_config.lstm_units * random.choice([0.8, 1.0, 1.2]))
        elif mutation_choice == "dense_units":
            next_config.dense_units = int(next_config.dense_units * random.choice([0.8, 1.0, 1.2]))

        self._evolve_recursive(next_config, generations_left - 1, max_generations)

    def evolve(self, max_generations: int = 100):
        logger.info(f"Starting recursive evolution for {max_generations} generations")
        initial_config = NeuroBridgeConfig()
        self._evolve_recursive(initial_config, max_generations, max_generations)
        return self.best_config
