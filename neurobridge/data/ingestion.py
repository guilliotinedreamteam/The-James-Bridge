import mne
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ECoGIngestionPipeline:
    """
    Ingestion pipeline strictly for legitimate, publicly released medical ECoG/EEG datasets.
    Supported formats: EDF (European Data Format), BDF (BioSemi Data Format).
    """
    def __init__(self, expected_channels: int = 128):
        self.expected_channels = expected_channels

    def load_medical_dataset(self, file_path: str) -> mne.io.Raw:
        """
        Loads a standard clinical EEG/ECoG file.
        Returns an MNE Raw object containing the continuous data and annotations.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Medical data file not found at: {file_path}")

        logger.info(f"Initiating load of medical data: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.edf':
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose='ERROR')
            elif ext == '.bdf':
                raw = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')
            elif ext == '.vhdr':
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose='ERROR')
            else:
                raise ValueError(f"Unsupported medical file format: {ext}. Only .edf, .bdf, or .vhdr are permitted.")
            
            logger.info(f"Successfully loaded dataset. Sampling frequency: {raw.info['sfreq']} Hz, Channels: {raw.info['nchan']}")
            
            if raw.info['nchan'] < self.expected_channels:
                logger.warning(f"Dataset has {raw.info['nchan']} channels, expected at least {self.expected_channels}.")
                
            return raw

        except Exception as e:
            logger.error(f"Failed to ingest medical dataset: {str(e)}")
            raise e

    def extract_numpy_arrays(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extracts the raw continuous data into a (channels, times) numpy array for downstream processing.
        """
        # MNE get_data returns shape (n_channels, n_times)
        data = raw.get_data()
        logger.info(f"Extracted numpy array with shape: {data.shape}")
        return data
