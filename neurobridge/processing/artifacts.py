import logging
import numpy as np
from scipy.signal import butter, lfilter, iirnotch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArtifactRejector:
    """
    Phase 8: Signal Artifact Rejection.
    Filters raw ECoG/EEG signals to remove physiological and environmental noise.
    Prevents false actuation of prosthetics caused by non-neural voltage spikes.
    """
    def __init__(self, sfreq: int = 500):
        self.sfreq = sfreq

    def apply_bandpass(self, data: np.ndarray, lowcut: float = 1.0, highcut: float = 100.0, order: int = 4) -> np.ndarray:
        """
        Applies a Butterworth bandpass filter to preserve neural-relevant frequencies.
        """
        logger.info(f"Applying Bandpass Filter: {lowcut}-{highcut} Hz")
        nyq = 0.5 * self.sfreq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data, axis=-1)

    def apply_notch(self, data: np.ndarray, freq: float = 60.0, quality: float = 30.0) -> np.ndarray:
        """
        Applies a notch filter to eliminate power line interference (50/60Hz).
        """
        logger.info(f"Applying Notch Filter at {freq} Hz")
        nyq = 0.5 * self.sfreq
        w0 = freq / nyq
        b, a = iirnotch(w0, quality)
        return lfilter(b, a, data, axis=-1)

    def threshold_rejection(self, data: np.ndarray, microvolt_limit: float = 200.0) -> np.ndarray:
        """
        Amplitude Thresholding: Discards or zeroes out segments with massive spikes.
        Standard for removing EMG (muscle) or electrode movement artifacts.
        """
        # data expected shape (channels, samples)
        logger.info(f"Applying Amplitude Rejection: Limit = {microvolt_limit} uV")
        mask = np.abs(data) > microvolt_limit
        
        # Count artifact events
        artifact_count = np.sum(mask)
        if artifact_count > 0:
            logger.warning(f"Detected {artifact_count} artifact samples exceeding threshold. Zeroing out signal.")
            data[mask] = 0.0
            
        return data

    def full_clinical_clean(self, data: np.ndarray) -> np.ndarray:
        """
        Executes the full clinical cleaning pipeline.
        """
        logger.info("Executing Full Clinical Cleaning Pipeline (Phase 8)")
        d1 = self.apply_notch(data, freq=60.0) # Notch US power line
        d2 = self.apply_bandpass(d1, lowcut=1.0, highcut=150.0) # Broad bandpass
        d3 = self.threshold_rejection(d2, microvolt_limit=250.0) # Kill movement noise
        return d3
