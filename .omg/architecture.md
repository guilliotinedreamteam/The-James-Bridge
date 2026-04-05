# Phase 3: Systems Engineering Blueprint - Neurobridge V1

## 1. Hardware-to-Software Boundaries

As defined in the V1 PRD, direct integration with physical ECoG medical hardware is out of scope. The system operates on pre-recorded or simulated data, establishing clear boundaries:

*   **Ingestion Boundary:** The software pipeline begins at the data loader or REST API endpoint, expecting 128-channel high-density ECoG arrays.
*   **Signal Adaptation Layer:** The raw signal (assumed to be originally at 1000 Hz) enters the preprocessing module where it is immediately downsampled to the 100 Hz working frequency.
*   **Normalization:** Z-score normalization is applied across the 128 feature channels to standardize the input distribution before it reaches the decoding models.
*   **Sequence Shaping:** Data is reshaped via zero-padding or trimming to fit the fixed 100-timestep windows (`(batch_size, 100, 128)`) for offline training, or sliced into `(1, 1, 128)` tensors for real-time inference.

## 2. Neural Decoding Architecture

The core of the system is split into two distinct execution paths to balance training capacity with inference speed:

*   **Offline Training Architecture (High Context):** A Bidirectional LSTM model implemented in TensorFlow/Keras. It consumes sequences of shape `(batch_size, 100, 128)` and predicts 41-class one-hot encoded phoneme targets. The bidirectional nature allows the model to leverage both past and future context for high-accuracy weight initialization.
*   **Online Inference Architecture (Low Latency):** A Unidirectional LSTM decoder. This model transfers learned representations from the offline model but processes data continuously as single frames (`timesteps=1`), strictly mapping the `(1, 1, 128)` input into phoneme probabilities without look-ahead delays.

## 3. Telemetry Pipelines

To ensure system observability and model performance tracking:

*   **Training & Evaluation Telemetry:** The `train` and `evaluate` CLI commands output and log categorical cross-entropy loss and frame-wise phoneme accuracy.
*   **Inference Telemetry:** The REST API (`serve`) tracks incoming HTTP POST requests, logging request rates, payload validation states, inference execution times, and the resulting Top-K phoneme distributions.
*   **System State:** The `info` CLI command acts as a localized diagnostic probe, detailing current system capabilities, loaded model weights, and hyperparameter configurations.

## 4. Latency Tradeoffs

The system architecture makes deliberate tradeoffs to achieve real-time Brain-Computer Interface (BCI) performance:

*   **Unidirectional vs. Bidirectional LSTM:** The shift from a Bidirectional model (offline) to a Unidirectional model (online) intentionally sacrifices the accuracy gains of future-context processing. This is a necessary tradeoff to achieve the near-zero latency required for real-time speech synthesis.
*   **Downsampling (1000 Hz to 100 Hz):** We trade high-frequency neural temporal resolution for a 10x reduction in computational load and memory bandwidth. 100 Hz provides sufficient temporal granularity for phoneme transition detection while keeping inference times well within the real-time budget.
*   **Fixed Sequence Alignment:** Avoiding complex dynamic sequence alignment models (like CTC or DTW) in V1 trades alignment flexibility for deterministic, predictable inference execution times.

## 5. Data Security Structures

While V1 operates primarily on mock or pre-recorded data, foundational security and data handling structures must be established:

*   **Subject Data Isolation:** The pipeline assumes subject-specific training. Model weights and pre-recorded datasets must be logically partitioned per subject to prevent cross-contamination of neural profiles.
*   **API Payload Validation:** The `serve` REST API must enforce strict tensor shape and type validation (e.g., verifying input arrays exactly match `(1, 1, 128)`) before passing data to the TensorFlow backend. This prevents application crashes or malicious payloads from exploiting the inference engine.
*   **Stateless Request Handling:** The inference server should handle HTTP requests statelessly where possible. While the Unidirectional LSTM must maintain an internal state across consecutive frames of a single session, this state must be explicitly managed, scoped to individual inference sessions, and securely flushed upon session termination.
