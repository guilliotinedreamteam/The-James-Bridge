import numpy as np
import logging
from scipy.signal import resample
from scipy.stats import zscore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Phase 2: Core signal processing pipeline for high-density ECoG medical data.
    Standardizes clinical data into LSTM-ready sequence tensors.
    Integrated with Phase 8 Artifact Rejection.
    """
    def __init__(self, original_freq: int = 1000, target_freq: int = 100, target_timesteps: int = 100):
        self.original_freq = original_freq
        self.target_freq = target_freq
        self.target_timesteps = target_timesteps
        logger.info(f"Initialized SignalProcessor (Target Freq: {self.target_freq}Hz, Timesteps: {self.target_timesteps})")

    def apply_artifact_rejection(self, data: np.ndarray, sfreq: int) -> np.ndarray:
        """
        Phase 8: Cleans the signal of noise and rejects artifacts.
        """
        from neurobridge.processing.artifacts import ArtifactRejector
        rejector = ArtifactRejector(sfreq=sfreq)
        return rejector.full_clinical_clean(data)

    def extract_high_gamma(self, data: np.ndarray, sfreq: int) -> np.ndarray:
        """
        Extracts the High-Gamma analytic amplitude envelope (70-150Hz).
        This is the gold-standard feature for ECoG speech decoding.
        """
        from scipy.signal import butter, filtfilt, hilbert
        
        # Nyquist frequency is half the sampling rate
        nyq = 0.5 * sfreq
        
        # If the sampling rate is too low to capture High-Gamma, return original data
        if sfreq <= 150 * 2:
            logger.warning(f"Sampling frequency {sfreq}Hz is too low for 150Hz High-Gamma extraction. Skipping.")
            return data
            
        logger.info("Extracting High-Gamma (70-150Hz) analytic amplitude envelope.")
        
        # 1. Bandpass filter 70-150Hz
        low = 70.0 / nyq
        high = 150.0 / nyq
        b, a = butter(4, [low, high], btype='band')
        
        # filtfilt applies the filter forward and backward for zero phase distortion
        filtered = filtfilt(b, a, data, axis=-1)
        
        # 2. Extract analytic amplitude (envelope) via Hilbert transform
        # np.abs of hilbert transform gives the envelope power
        analytic_signal = hilbert(filtered, axis=-1)
        amplitude_envelope = np.abs(analytic_signal)
        
        return amplitude_envelope

    def downsample_signals(self, data: np.ndarray, current_freq: int = None) -> np.ndarray:
        """
        Downsamples the raw ECoG signal to meet real-time inference latency budgets.
        Integrated with Phase 8 cleaning and High-Gamma feature extraction.
        """
        if current_freq is None:
            current_freq = self.original_freq

        # 1. Apply artifact rejection
        data = self.apply_artifact_rejection(data, sfreq=current_freq)
        
        # 2. Extract High-Gamma envelope BEFORE downsampling (so we don't alias away the high frequencies)
        data = self.extract_high_gamma(data, sfreq=current_freq)

        if current_freq == self.target_freq:
            logger.info("Signal is already at target frequency. Skipping downsampling.")
            return data
            
        logger.info(f"Downsampling signals from {current_freq} Hz to {self.target_freq} Hz")
        
        # Calculate the number of samples in the new sequence
        time_axis = -1 
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
