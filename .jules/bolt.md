## 2024-06-11 - [TensorFlow Prediction Overhead for Real-Time Inference]
**Learning:** `model.predict()` in TensorFlow/Keras adds massive overhead due to its batch processing setup, making it unsuitable for real-time single-item inference.
**Action:** Always use direct model invocation `model(inputs, training=False).numpy()` for latency-sensitive, single-item prediction tasks.

## 2024-06-11 - [NumPy Operations on Python Lists]
**Learning:** Executing NumPy operations like `np.argmax` or `np.max` on Python lists incurs significant overhead due to implicit array conversions.
**Action:** Always perform NumPy operations on native NumPy arrays before converting to lists for JSON serialization or other operations requiring Python lists.
