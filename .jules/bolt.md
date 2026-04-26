## 2024-06-11 - [TensorFlow Prediction Overhead for Real-Time Inference]
**Learning:** `model.predict()` in TensorFlow/Keras adds massive overhead due to its batch processing setup, making it unsuitable for real-time single-item inference.
**Action:** Always use direct model invocation `model(inputs, training=False).numpy()` for latency-sensitive, single-item prediction tasks.

## 2024-05-15 - [NumPy Implicit Conversion Overhead]
**Learning:** Performing operations like `np.argmax()` or `np.max()` on Python lists causes massive performance overhead due to implicit, on-the-fly array conversions.
**Action:** Always perform matrix math and reduction operations on the native `numpy` arrays before converting them to Python lists (e.g. `tolist()`) for JSON serialization.
