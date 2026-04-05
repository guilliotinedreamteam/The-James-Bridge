import numpy as np
import logging
from scipy.signal import resample
from scipy.stats import zscore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Phase 2: Core signal processing pipeline for high-density ECoG medical data.
    Executes standard BCI preprocessing: Downsampling, Z-score Normalization, and Sequence Shaping.
    """
    def __init__(self, original_freq: int = 1000, target_freq: int = 100, target_timesteps: int = 100):
        self.original_freq = original_freq
        self.target_freq = target_freq
        self.target_timesteps = target_timesteps
        logger.info(f"Initialized SignalProcessor (Target Freq: {self.target_freq}Hz, Timesteps: {self.target_timesteps})")

    def downsample_signals(self, data: np.ndarray, current_freq: int = None) -> np.ndarray:
        """
        Downsamples the raw ECoG signal to meet real-time inference latency budgets.
        Expects data shape: (channels, times) or (batch, channels, times)
        Returns: Downsampled data array.
        """
        if current_freq is None:
            current_freq = self.original_freq

        if current_freq == self.target_freq:
            logger.info("Signal is already at target frequency. Skipping downsampling.")
            return data
            
        logger.info(f"Downsampling signals from {current_freq} Hz to {self.target_freq} Hz")
        
        # Calculate the number of samples in the new sequence
        time_axis = -1 # Assuming time is always the last axis (channels, times)
        num_samples = int(data.shape[time_axis] * (self.target_freq / current_freq))
        
        downsampled_data = resample(data, num_samples, axis=time_axis)
        return downsampled_data

    def z_score_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Applies robust Z-score normalization: (data - mean) / std.
        Normalized per channel to prevent one active electrode from washing out the others.
        Expects data shape: (channels, times)
        """
        logger.info("Applying Z-score normalization across the time axis (per channel).")
        # Ensure we don't divide by zero if a channel is completely dead
        epsilon = 1e-10
        means = np.mean(data, axis=-1, keepdims=True)
        stds = np.std(data, axis=-1, keepdims=True)
        normalized_data = (data - means) / (stds + epsilon)
        return normalized_data

    def shape_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Shapes the continuous time-series array into fixed chunks of `target_timesteps` for LSTM ingestion.
        Expects data shape: (channels, times)
        Returns data shape: (batch_size, timesteps, channels)
        """
        logger.info(f"Shaping continuous sequence into blocks of {self.target_timesteps} timesteps.")
        
        channels, total_time = data.shape
        # Transpose to (times, channels) for slicing
        data_t = data.T 
        
        # Calculate how many full blocks we can extract
        num_blocks = total_time // self.target_timesteps
        
        if num_blocks == 0:
            raise ValueError(f"Not enough data to create even one sequence of {self.target_timesteps} timesteps.")
            
        # Truncate any remainder
        truncated_data = data_t[:num_blocks * self.target_timesteps, :]
        
        # Reshape to (batch_size, timesteps, channels)
        reshaped_data = truncated_data.reshape((num_blocks, self.target_timesteps, channels))
        
        logger.info(f"Final shaped tensor for neural decoding: {reshaped_data.shape}")
        return reshaped_data

    def align_targets(self, phoneme_ids: np.ndarray, num_classes: int = 41) -> np.ndarray:
        """
        Maps sparse phoneme IDs to a one-hot encoded matrix for Categorical Crossentropy.
        """
        logger.info(f"Aligning sparse phoneme targets to {num_classes}-class one-hot vectors.")
        one_hot = np.eye(num_classes)[phoneme_ids]
        return one_hot
