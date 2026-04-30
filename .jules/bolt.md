## 2024-06-11 - [TensorFlow Prediction Overhead for Real-Time Inference]
**Learning:** `model.predict()` in TensorFlow/Keras adds massive overhead due to its batch processing setup, making it unsuitable for real-time single-item inference.
**Action:** Always use direct model invocation `model(inputs, training=False).numpy()` for latency-sensitive, single-item prediction tasks.

## 2024-06-12 - [NumPy Overhead from Implicit List Array Conversion]
**Learning:** Performing operations like `np.argmax()` or `np.max()` directly on Python lists causes massive implicit array conversion overhead. This bottleneck is crucial when processing elements frequently, such as in real-time server prediction endpoints.
**Action:** Always perform NumPy operations directly on the native `np.ndarray` objects before casting them to `.tolist()` for JSON serialization.
