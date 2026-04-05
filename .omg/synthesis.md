# Phase 1: Synthesis Phase - Neurobridge Project

This document synthesizes the core technical requirements, signal processing logic, and prosthetic integration goals extracted from the `neurobridge.py` CLI script and `neurobridge_py.ipynb` notebook.

## 1. Pure Technical Requirements

### Core Neural Network Architecture
*   **Framework:** TensorFlow / Keras.
*   **Input Representation:** 3D tensor representing `(batch_size, timesteps, neural_features)`.
    *   `NUM_TIMESTEPS`: 100 (e.g., 100ms sequence of data).
    *   `NUM_FEATURES`: 128 (number of high-density ECoG electrode channels).
*   **Hidden Layers:**
    *   **Offline/Training Decoder:** Uses `Bidirectional(LSTM(256, return_sequences=True))` to capture bidirectional context.
    *   **Real-time/Inference Decoder:** Uses unidirectional `LSTM(256, return_sequences=True)` with `timesteps=1` to allow continuous single-frame inference.
    *   Each recurrent layer is followed by `BatchNormalization()`.
    *   Intermediate projection via `TimeDistributed(Dense(128, activation='relu'))`.
*   **Output Layer:** `TimeDistributed(Dense(NUM_PHONEMES, activation='softmax'))`.
    *   `NUM_PHONEMES`: 41 (40 linguistic phonemes + 1 silence token).
*   **Compilation:**
    *   Optimizer: `Adam`.
    *   Loss Function: `categorical_crossentropy`.

### Command-Line Interface (CLI)
*   **Entry Point:** `neurobridge.py`.
*   **Commands:**
    *   `train`: Trigger `train_model(epochs, batch_size)`.
    *   `evaluate`: Trigger `evaluate_model(num_test_samples)` tracking final accuracy and loss.
    *   `serve`: Expose REST API server via `start_server(port, debug)`.
    *   `predict`: Perform inference using `RealtimeDecoder` for Top-K phonemes prediction.
    *   `info`: Display model configuration and system details (e.g., GPU availability).

## 2. Signal Processing Logic

### Data Preprocessing
*   **Signal Normalization:** ECoG data arrays are handled as `float32`. Required preprocessing includes Z-score normalization (`(data - mean) / std`) across the feature channels.
*   **Downsampling:** Reduction of high-frequency raw ECoG signals (e.g., 1000 Hz) to a target working frequency (e.g., 100 Hz).
*   **Sequence Shaping:** Time-series arrays are either zero-padded or trimmed to strictly match the configured `timesteps` bounds.

### Target Alignment
*   **Phoneme Labeling:** Raw phoneme IDs (sparse, integer-based) are mapped directly to a 41-class one-hot encoded representation via `tf.keras.utils.to_categorical`.
*   **Sequence Matching:** Timesteps between ECoG samples and phoneme sequence targets must be strictly aligned, necessitating techniques like Connectionist Temporal Classification (CTC) preparation or Dynamic Time Warping (DTW) for real-world unsegmented labels.

## 3. Prosthetic Integration Goals

### Neuroprosthetic Objectives
*   **Brain-Computer Interface (BCI) Purpose:** Translate direct neural activity (ECoG) into sequences of fundamental speech units (phonemes) to restore natural speech capabilities.
*   **Real-Time Processing:** The system architecture fundamentally bifurcates into an offline bidirectional trainer and an online unidirectional real-time decoder. The online decoder evaluates 1 ECoG frame at a time (`timesteps=1`) to emit phoneme probabilities instantaneously.

### Synthesizer Integration
*   **Vocoder Pipeline:** Phoneme ID arrays emitted by the decoder must be mapped to conceptual phoneme strings.
*   **Audible Speech Generation:** The ultimate prosthetic goal requires a bridging module (`synthesize_speech_from_phonemes`) that converts generated phoneme sequences into physical, audible speech waveforms using text-to-speech (TTS) engines or neural vocoders (e.g., evaluating duration, sampling rate).

### Evaluation & Metrics
*   **Phoneme-Level Accuracy:** Frame-wise accuracy calculation is the primary metric, comparing the argmax of predicted probability arrays against true phoneme IDs per timestep.
