# Neurobridge V1: Master Technical Blueprint

## 1. Executive Summary
Neurobridge V1 is a Brain-Computer Interface (BCI) software pipeline. It translates high-density 128-channel ECoG neural activity into sequences of 41 fundamental speech units (phonemes) for real-time speech synthesis. This blueprint defines the architecture, data pipeline, and execution strategy for V1.

## 2. System Boundaries & Non-Goals
**In-Scope (V1):**
* Offline Bidirectional LSTM training pipeline (context-rich).
* Online Unidirectional LSTM inference decoder (low-latency, `timesteps=1`).
* Preprocessing: Z-score normalization, downsampling (1000 Hz → 100 Hz), sequence shaping.
* CLI/API (`train`, `evaluate`, `predict`, `serve`, `info`).

**Out-of-Scope (V1):**
* Direct hardware integration (operates on pre-recorded/simulated data).
* Production-grade Neural Vocoder / TTS.
* Dynamic sequence alignment (CTC/DTW). Sequence length is fixed (`NUM_TIMESTEPS=100`).
* Multi-patient generalization (subject-specific only).
* GUI/Frontend dashboards.

## 3. Core Architecture
The system bifurcates to balance offline capacity with online real-time constraints:

*   **Offline Decoder (Training):**
    *   **Input:** `(batch_size, 100, 128)`
    *   **Architecture:** `Bidirectional(LSTM(256, return_sequences=True))` -> `BatchNormalization()` -> `TimeDistributed(Dense(128, relu))` -> `TimeDistributed(Dense(41, softmax))`.
    *   **Loss/Optimizer:** Categorical Crossentropy / Adam.
*   **Online Decoder (Inference):**
    *   **Input:** `(1, 1, 128)`
    *   **Architecture:** Unidirectional `LSTM(256, return_sequences=True)` mapping single frames to phoneme probabilities with near-zero look-ahead delay. Transfers learned weights from the offline model.
    *   **State Management:** HTTP stateless handling with explicit session-scoped LSTM state management during continuous single-frame inference.

## 4. Signal Processing Pipeline
*   **Ingestion:** Expects 128-channel `float32` ECoG arrays.
*   **Downsampling:** Reduces 1000 Hz raw signals to 100 Hz to meet real-time inference budgets (computational reduction).
*   **Normalization:** Z-score normalization `(data - mean) / std` across feature channels.
*   **Sequence Shaping:** Zero-padding/trimming to strict `timesteps` bounds.
*   **Target Alignment:** Maps sparse phoneme IDs to 41-class one-hot encoded arrays.

## 5. Interfaces, Security & Telemetry
*   **CLI (`neurobridge.py`):**
    *   `train`: Execute training loop and persist subject-specific weights.
    *   `evaluate`: Compute frame-wise phoneme accuracy and loss.
    *   `predict`: CLI-based Top-K phoneme inference.
    *   `info`: System diagnostics and hardware capabilities.
    *   `serve`: Expose REST API for inference.
*   **Security & Validation:**
    *   Strict input tensor shape/type validation in the REST API.
    *   Subject-specific model weight partitioning to prevent data cross-contamination.
*   **Telemetry:** Logs inference execution times, request rates, Top-K distributions, and offline metrics.

## 6. Execution Plan & Milestones
*   **Phase 1: Foundation & Data Ingestion.** Initialize repository, build CLI router, implement `info` command, and build robust Medical Data Ingestion pipelines capable of reading legitimate publicly released high-density ECoG datasets (e.g., standard `.edf` or `.nwb` formats used in real clinical trials). No mock or fake data allowed.
*   **Phase 2: Signal Processing.** Implement downsampling, Z-score normalization, sequence shaping, and phoneme one-hot target alignment.
*   **Phase 3: Neural Decoding.** Construct both Bidirectional (Offline) and Unidirectional (Online) LSTM architectures in TensorFlow/Keras.
*   **Phase 4: Training & Evaluation.** Hook Offline model to CLI. Implement `train` (weight persistence) and `evaluate` (frame-wise accuracy calculation).
*   **Phase 5: Real-Time Serving.** Hook Online model to CLI/API. Implement `predict` and `serve` with strict payload validation and session-scoped state management.