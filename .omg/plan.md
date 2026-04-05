# Phase 4: Project Phasing & Execution Plan - Neurobridge V1

Based on the PRD and Architecture documents, this plan breaks down the implementation of Neurobridge V1 into ordered, actionable phases with exact validation checkpoints.

## Phase 1: Project Skeleton & Mock Data Foundation
**Goal:** Establish the foundational repository structure, the CLI routing, and the mock data generation necessary for subsequent phases without relying on actual ECoG hardware.

*   **Tasks:**
    1. Initialize the Python project and define dependencies (TensorFlow/Keras, NumPy, Flask/FastAPI for API).
    2. Implement `neurobridge.py` as the main CLI entry point using `argparse` or `click`.
    3. Scaffold the CLI commands: `train`, `evaluate`, `predict`, `serve`, `info`.
    4. Implement the `info` command to print system configuration and diagnostic state.
    5. Build the Mock Data Generator to produce simulated ECoG arrays of shape `(batch_size, 100, 128)` for offline and `(1, 1, 128)` for online, as well as 41-class mock phoneme targets.
*   **Validation Checkpoints:**
    *   [ ] `python neurobridge.py info` successfully runs and outputs the current system state.
    *   [ ] Calling the mock data generator successfully yields tensors of exact expected shapes without errors.
    *   [ ] Unit tests for the mock generator are passing.

## Phase 2: Signal Processing Pipeline
**Goal:** Implement the data transformation functions required to bridge the gap between raw hardware signals and the neural decoder inputs.

*   **Tasks:**
    1. Implement the downsampling module to reduce 1000 Hz signal data to the 100 Hz working frequency.
    2. Implement Z-score normalization to standardize the 128 feature channels.
    3. Implement sequence shaping functions (zero-padding/trimming) to guarantee fixed 100-timestep windows.
    4. Implement the target alignment module to convert phoneme IDs into a 41-class one-hot encoded array.
*   **Validation Checkpoints:**
    *   [ ] Unit tests confirm that Z-score normalization results in a mean of ~0 and standard deviation of ~1 across feature channels.
    *   [ ] Unit tests confirm the downsampling function correctly scales the time dimension by a factor of 10.
    *   [ ] Unit tests confirm target alignment produces `(batch_size, 100, 41)` shaped arrays.

## Phase 3: Neural Decoding Architecture
**Goal:** Define and compile the core machine learning models in TensorFlow/Keras.

*   **Tasks:**
    1. Construct the **Offline Training Architecture**: A Bidirectional LSTM model that accepts `(batch_size, 100, 128)` and outputs `(batch_size, 100, 41)`.
    2. Construct the **Online Inference Architecture**: A Unidirectional LSTM decoder that accepts `(1, 1, 128)` and outputs a `(1, 1, 41)` phoneme probability distribution.
    3. Implement the logic to transfer trained weights from the Bidirectional model to the Unidirectional model where applicable.
*   **Validation Checkpoints:**
    *   [ ] Both the Bidirectional and Unidirectional models successfully compile in TensorFlow/Keras without shape mismatches.
    *   [ ] A forward pass of mock data through the untrained Offline model returns the expected output shape `(batch_size, 100, 41)`.
    *   [ ] A forward pass of mock data through the untrained Online model returns the expected output shape `(1, 1, 41)`.

## Phase 4: Training & Evaluation Execution
**Goal:** Hook up the Offline model to the CLI to enable training and performance measurement.

*   **Tasks:**
    1. Implement the logic for the `train` command: Load mock dataset, compile the Offline model with categorical cross-entropy loss, execute the training loop, and save the resulting model weights to disk subject-specifically.
    2. Implement telemetry logging during training (loss and metrics).
    3. Implement the logic for the `evaluate` command: Load saved model weights, run inference over a test dataset, and compute/log frame-wise phoneme accuracy.
*   **Validation Checkpoints:**
    *   [ ] Executing `python neurobridge.py train` runs the training loop, logs metrics, and successfully writes a model weights file (e.g., `.h5` or `.keras`) to disk.
    *   [ ] Executing `python neurobridge.py evaluate` successfully loads the weights, processes test data, and logs the calculated accuracy and loss.

## Phase 5: Real-Time Inference & REST API Serving
**Goal:** Expose the low-latency Unidirectional LSTM model via a local REST API and CLI prediction command.

*   **Tasks:**
    1. Implement the logic for the `predict` CLI command: Load weights into the Unidirectional LSTM, process a single `(1, 1, 128)` frame, and output the Top-K predicted phonemes.
    2. Build the REST API for the `serve` command using a lightweight web framework.
    3. Implement strict payload validation in the API to enforce `(1, 1, 128)` input shapes.
    4. Implement stateful request handling that maintains LSTM state for consecutive frames but safely isolates/flushes state per inference session.
*   **Validation Checkpoints:**
    *   [ ] Executing `python neurobridge.py predict` prints the Top-K phonemes for a provided mock tensor.
    *   [ ] Executing `python neurobridge.py serve` successfully starts the HTTP server.
    *   [ ] Sending a valid HTTP POST request with a `(1, 1, 128)` tensor returns a successful JSON response with phoneme probabilities.
    *   [ ] Sending an HTTP POST request with an invalid tensor shape is successfully rejected with an HTTP 400 Bad Request status.
