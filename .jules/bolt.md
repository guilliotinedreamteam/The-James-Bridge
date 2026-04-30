## 2024-06-11 - [TensorFlow Prediction Overhead for Real-Time Inference]
**Learning:** `model.predict()` in TensorFlow/Keras adds massive overhead due to its batch processing setup, making it unsuitable for real-time single-item inference.
**Action:** Always use direct model invocation `model(inputs, training=False).numpy()` for latency-sensitive, single-item prediction tasks.

## 2024-06-25 - [NumPy Overhead from Implicit List Conversions]
**Learning:** Performing operations like `np.argmax` or `np.max` on a standard Python list causes NumPy to implicitly convert the list back into an array first, which adds significant processing overhead inside hot paths.
**Action:** Always complete native array operations (like `.max()`, `.argmax()`) directly on the `np.ndarray` before converting to a list using `.tolist()` for JSON serialization.
