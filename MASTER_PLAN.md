# James Bridge V1: Master Technical Blueprint

## 1. Executive Summary
James Bridge V1 is an enterprise-grade Brain-Computer Interface (BCI) software pipeline. It translates high-density 128-channel ECoG neural activity into sequences of 41 fundamental speech units (phonemes) for real-time speech synthesis. This blueprint defines the architecture, data pipeline, and execution strategy for V1.

## 2. System Boundaries & Non-Goals
**In-Scope (V1):**
* Offline Bidirectional LSTM training pipeline (context-rich).
* Online Unidirectional LSTM inference decoder (low-latency, `timesteps=1`).
* Preprocessing: Z-score normalization, downsampling (1000 Hz → 100 Hz), sequence shaping.
* CLI/API Orchestration (`train`, `evaluate`, `predict`, `serve`, `info`).
* Sub-100ms pipeline execution with 0.17ms model inference.

**Out-of-Scope (V1):**
* Direct bare-metal microcontroller inference (handled via high-concurrency local server until Phase 4).
* Production-grade Neural Vocoder / TTS.
* Multi-patient generalization (Subject-specific models only).

## 3. Core Architecture
The system bifurcates to balance offline capacity with online real-time constraints, utilizing a Hybrid CNN-LSTM to capture both spatial and temporal features:

* **Offline Decoder (Training):**
    * **Input:** `(batch_size, 100, 128)`
    * **Spatial Filter:** `Conv1D(filters, kernel_size)` -> `BatchNormalization()` -> `MaxPooling1D()`
    * **Temporal Filter:** `Bidirectional(LSTM(256, return_sequences=True))` 
    * **Dense Mapping:** `TimeDistributed(Dense(128, relu))` -> `TimeDistributed(Dense(41, softmax))`

* **Online Decoder (Real-Time Inference):**
    * **Input:** `(1, 1, 128)` (Streaming frame-by-frame)
    * **Spatial Filter:** `Conv1D` (State-preserved)
    * **Temporal Filter:** `LSTM(256, stateful=True, return_sequences=False)` (Context is maintained in the internal hidden states $h_t, c_t$).
    * **Optimization:** Direct tensor invocation bypassing `.predict()` overhead to achieve ~0.17ms execution.

## 4. Streaming Data Ingestion (The S3 Pipeline)
Due to the massive footprint of clinical datasets (e.g., OpenNeuro `ds003620`), the pipeline enforces a **Streaming Enrollment Pattern**:
1. Download patient data via AWS S3 CLI.
2. Extract ECoG matrices and align phoneme targets.
3. Train subject-specific architecture.
4. Finalize weights and instantly purge raw patient files to reclaim inodes and disk space.

## 5. Interface & Actuation (Phase 7)
* **Actuation Flow:** REST API -> Threshold Validation -> Hardware Command Dispatch.
* **Safety Envelope:** 100% test coverage on simulated endpoint hits. A command is only dispatched if the neural confidence exceeds a strict predictive threshold.
