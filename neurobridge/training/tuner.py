import logging
import numpy as np
from neurobridge.model.decoder import NeurobridgeDecoder
from neurobridge.training.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperTuner:
    """
    Tuning Suite for Neurobridge BCI models using legitimate clinical data.
    Executes hyperparameter searches to optimize the LSTM mapping.
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = x_data
        self.y_data = y_data
        
    def execute_grid_search(self, lstm_options: list, dropout_options: list, epochs: int = 5):
        """
        Runs a grid search over specified LSTM units and Dropout rates.
        Outputs the performance metrics for each configuration on the clinical data.
        """
        results = []
        timesteps = self.x_data.shape[1]
        channels = self.x_data.shape[2]
        
        logger.info(f"Starting Hyperparameter Grid Search on legitimate clinical dataset.")
        
        for units in lstm_options:
            for dropout in dropout_options:
                logger.info(f"--- Tuning Run: LSTM Units={units}, Dropout={dropout} ---")
                
                decoder = NeurobridgeDecoder(
                    timesteps=timesteps, 
                    channels=channels, 
                    lstm_units=units, 
                    dropout_rate=dropout
                )
                model = decoder.build_offline_decoder()
                
                trainer = ModelTrainer(model=model)
                history = trainer.train(
                    x_train=self.x_data, 
                    y_train=self.y_data, 
                    epochs=epochs, 
                    batch_size=8,
                    model_name=f"tune_u{units}_d{dropout}"
                )
                
                final_acc = history.history['accuracy'][-1]
                final_loss = history.history['loss'][-1]
                
                results.append({
                    "units": units,
                    "dropout": dropout,
                    "accuracy": final_acc,
                    "loss": final_loss
                })
                
        logger.info("Grid Search Complete. Tuning Results:")
        for res in results:
            logger.info(f"Config [Units: {res['units']}, Dropout: {res['dropout']}] -> Acc: {res['accuracy']:.4f}, Loss: {res['loss']:.4f}")
            
        return results
