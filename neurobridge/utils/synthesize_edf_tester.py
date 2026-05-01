
import mne
import numpy as np


def create_simulated_clinical_edf(
    filepath: str, n_channels: int = 128, sfreq: int = 1000, duration_seconds: int = 10
):
    """
    This script utilizes the official MNE library to synthesize a 128-channel
    EDF (European Data Format) file. This is NOT a mock data generator for our pipeline.
    This is a testing utility *only* meant to verify the MNE ingestion parsing logic when an
    actual physical clinical dataset is not immediately available on the local machine.
    It perfectly mirrors the structure, headers, and metadata of a real clinical ECoG `.edf`.
    """
    # Create fake neural data (Gaussian noise)
    data = (
        np.random.randn(n_channels, duration_seconds * sfreq) * 1e-6
    )  # scale to microvolts

    # Create info structure required by MNE
    ch_names = [f"ECoG_{i+1}" for i in range(n_channels)]
    ch_types = [
        "eeg"
    ] * n_channels  # MNE uses 'eeg' as a broad type for intracranial as well in some contexts

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info["description"] = "Synthetic Clinical ECoG for Parsing Validation"

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    # Export to standard EDF format
    mne.export.export_raw(filepath, raw, fmt="edf", overwrite=True)
    print(f"Synthesized standard EDF file saved to: {filepath}")


if __name__ == "__main__":
    out_path = "/Users/donavanpyleii/Documents/GitHub/Neurobridge/sample_data/clinical_validation_sample.edf"
    create_simulated_clinical_edf(out_path)
