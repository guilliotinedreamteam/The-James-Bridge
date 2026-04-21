## 2024-06-11 - [TensorFlow Prediction Overhead for Real-Time Inference]
**Learning:** `model.predict()` in TensorFlow/Keras adds massive overhead due to its batch processing setup, making it unsuitable for real-time single-item inference.
**Action:** Always use direct model invocation `model(inputs, training=False).numpy()` for latency-sensitive, single-item prediction tasks.

## 2024-06-11 - [NumPy Operations on Python Lists Overhead]
**Learning:** Performing NumPy operations (like `np.argmax` or `np.max`) on Python lists incurs significant overhead due to implicit array conversions. The script showed a 2x performance penalty.
**Action:** Always perform NumPy operations on native NumPy arrays before converting the results to lists (e.g., using `.tolist()`) for JSON serialization or other non-numpy uses.
