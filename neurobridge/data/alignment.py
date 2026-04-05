import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelAligner:
    """
    Phase 4: Clinical Label Alignment Pipeline.
    Aligns the sparse events from clinical BIDS datasets (e.g., .tsv) to the dense, continuous 
    time-series ECoG arrays. Ensures the labels match the exact timesteps and batch dimensions 
    expected by the Bidirectional LSTM.
    """
    def __init__(self, original_sfreq: int, target_sfreq: int, target_timesteps: int, phoneme_classes: int = 41):
        self.original_sfreq = original_sfreq
        self.target_sfreq = target_sfreq
        self.target_timesteps = target_timesteps
        self.phoneme_classes = phoneme_classes
        # Ratio to scale sample indices from original frequency to downsampled target frequency
        self.downsample_ratio = self.target_sfreq / self.original_sfreq
        
        # A mock mapping dict since ds003620 is an auditory oddball task (S1, S2, etc.),
        # but our architecture is a Phoneme Decoder. We map these triggers into our 41-class 
        # phoneme space to prove the mathematical pipeline works on real clinical events.
        # In a real speech-decoding dataset, the .tsv 'trial_type' column would directly contain phonemes.
        self.trigger_to_phoneme = {
            'empty': 0,
            'S  1': 1,
            'S  2': 2,
            'S  3': 3,
            'S 10': 4,
            'S 20': 5,
        }

    def load_events(self, tsv_path: str) -> pd.DataFrame:
        """
        Loads the clinical events .tsv file.
        """
        logger.info(f"Loading clinical events from {tsv_path}")
        df = pd.read_csv(tsv_path, sep='\t')
        return df

    def align_to_tensor(self, df: pd.DataFrame, total_original_samples: int) -> np.ndarray:
        """
        Creates a continuous 1D array of phoneme classes parallel to the continuous neural array,
        then shapes it into (batch_size, timesteps) to match the Phase 2 sequence shaper.
        """
        logger.info("Aligning sparse clinical triggers to dense continuous time-series...")
        
        # Calculate total downsampled samples
        total_target_samples = int(total_original_samples * self.downsample_ratio)
        
        # Initialize continuous label array with 0 (silence/background)
        continuous_labels = np.zeros(total_target_samples, dtype=np.int32)
        
        # The ds003620 dataset records 'onset' in samples, not seconds, according to its metadata.
        for _, row in df.iterrows():
            # Extract sample onset and scale to the downsampled frequency
            original_sample = int(row['onset'])
            target_sample = int(original_sample * self.downsample_ratio)
            
            # Map the clinical trigger string to an integer
            trigger = str(row['trial_type']).strip()
            phoneme_id = self.trigger_to_phoneme.get(trigger, 0) # default to 0 if unknown
            
            # In a real speech task, duration matters. For oddball, the trigger is instantaneous.
            # We assign the phoneme ID to that exact sample timestep.
            if 0 <= target_sample < total_target_samples:
                continuous_labels[target_sample] = phoneme_id

        # Now, shape the continuous 1D array into blocks matching Phase 2: (num_blocks, target_timesteps)
        num_blocks = total_target_samples // self.target_timesteps
        truncated_labels = continuous_labels[:num_blocks * self.target_timesteps]
        reshaped_labels = truncated_labels.reshape((num_blocks, self.target_timesteps))
        
        logger.info(f"Labels aligned and shaped to: {reshaped_labels.shape}")
        
        # Finally, one-hot encode it for the LSTM output layer (batch, timesteps, 41)
        one_hot_targets = np.eye(self.phoneme_classes)[reshaped_labels]
        logger.info(f"One-hot target tensor generated: {one_hot_targets.shape}")
        
        return one_hot_targets
