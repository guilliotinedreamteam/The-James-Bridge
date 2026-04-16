## 2024-06-11 - [TensorFlow Prediction Overhead for Real-Time Inference]
**Learning:** `model.predict()` in TensorFlow/Keras adds massive overhead due to its batch processing setup, making it unsuitable for real-time single-item inference.
**Action:** Always use direct model invocation `model(inputs, training=False).numpy()` for latency-sensitive, single-item prediction tasks.

## 2024-06-11 - [NumPy Operations on Python Lists Overhead]
**Learning:** Performing `np.argmax` or `np.max` on a Python list invokes implicit conversion back to an array, causing significant overhead (often ~2x slower) compared to performing the operations on the native NumPy array before calling `.tolist()`.
**Action:** Always extract metrics (like argmax or max) from the native NumPy array *before* converting it to a Python list for JSON serialization.
