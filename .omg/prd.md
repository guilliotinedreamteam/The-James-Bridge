# Phase 2: PRD & Scope Definition - Neurobridge Project

## 1. Product Vision & Absolute Scope (V1)
Neurobridge V1 is a Brain-Computer Interface (BCI) software pipeline that translates direct neural activity (high-density ECoG signals) into sequences of fundamental speech units (phonemes) to enable real-time speech synthesis.

**In-Scope Features for V1:**
*   **Neural Decoding Architecture:**
    *   Implement an offline Bi-directional LSTM training pipeline using TensorFlow/Keras for sequence-to-sequence phoneme prediction from 128-channel ECoG data.
    *   Implement an online Unidirectional LSTM real-time inference decoder capable of processing continuous single-frame ECoG data (`timesteps=1`).
*   **Signal Processing Pipeline:**
    *   Data preprocessing module supporting Z-score normalization, signal downsampling (e.g., 1000 Hz to 100 Hz), and sequence shaping (zero-padding/trimming).
    *   Target alignment module for mapping 41-class one-hot encoded phoneme arrays and sequence matching.
*   **Command-Line Interface (CLI):**
    *   `train`: Functionality to train the offline model.
    *   `evaluate`: Functionality to compute frame-wise phoneme accuracy.
    *   `predict`: Functionality to perform Top-K phoneme prediction using the real-time decoder.
    *   `serve`: A REST API server for inference requests.
    *   `info`: System and model configuration display.

## 2. Non-Goals (Out of Scope for V1)
To ensure delivery and maintain focus, the following features are explicitly excluded from V1:
*   **Hardware Integration:** Direct interfacing with physical ECoG hardware or continuous streaming from medical devices. V1 will operate on pre-recorded or simulated offline data arrays.
*   **Production Vocoder / TTS:** A high-fidelity neural vocoder or complete, production-grade text-to-speech engine. V1 will output phoneme probabilities and sequences, but bridging to final audio waveforms is limited to a basic proof-of-concept script.
*   **Advanced Sequence Alignment Models:** Full Connectionist Temporal Classification (CTC) loss or Dynamic Time Warping (DTW) for unsegmented continuous speech labeling. V1 assumes pre-aligned or fixed-length sequences (`NUM_TIMESTEPS=100`).
*   **Multi-Patient Generalization:** Zero-shot transfer learning across multiple patients. V1 focuses on subject-specific model training and evaluation.
*   **GUI / Frontend Dashboard:** V1 is strictly a CLI and REST API backend. No visual dashboards for real-time brainwave monitoring will be built.

## 3. Ironclad Acceptance Criteria

### Model Training & Architecture
*   [ ] The codebase successfully compiles a Bidirectional LSTM training model and a Unidirectional LSTM inference model in TensorFlow/Keras.
*   [ ] The training pipeline successfully ingests input tensors of shape `(batch_size, 100, 128)` and outputs target tensors of shape `(batch_size, 100, 41)`.
*   [ ] The real-time inference model successfully processes single-timestep inputs `(1, 1, 128)` without sequence length errors.

### Signal Processing
*   [ ] Preprocessing functions correctly apply Z-score normalization across the 128 feature channels.
*   [ ] Downsampling functions accurately reduce high-frequency mock data to the target 100 Hz working frequency.
*   [ ] Phoneme IDs are accurately mapped to a 41-class one-hot categorical representation.

### CLI & API Execution
*   [ ] Executing `python neurobridge.py train` successfully runs the training loop and saves model weights.
*   [ ] Executing `python neurobridge.py evaluate` successfully computes and logs the categorical crossentropy loss and frame-wise accuracy.
*   [ ] Executing `python neurobridge.py serve` starts a functional REST API server that responds to HTTP POST requests with phoneme predictions.
*   [ ] Executing `python neurobridge.py predict` outputs Top-K predicted phonemes for a given input tensor.

### Testing & Validation
*   [ ] The repository includes a mock data generator to test the end-to-end pipeline without needing actual patient ECoG data.
*   [ ] Unit tests cover core preprocessing and model compilation steps to ensure deterministic behavior.