## 2024-06-11 - [TensorFlow Prediction Overhead for Real-Time Inference]
**Learning:** `model.predict()` in TensorFlow/Keras adds massive overhead due to its batch processing setup, making it unsuitable for real-time single-item inference.
**Action:** Always use direct model invocation `model(inputs, training=False).numpy()` for latency-sensitive, single-item prediction tasks.
## 2024-10-25 - Avoid Implicit List-to-Array Conversions
**Learning:** Calling NumPy operations like `np.argmax()` or `np.max()` on Python lists incurs significant overhead because NumPy implicitly converts the list back into a NumPy array first.
**Action:** Always perform NumPy operations on native NumPy arrays *before* converting them to lists for JSON serialization or other purposes.
