## 2024-06-11 - [TensorFlow Prediction Overhead for Real-Time Inference]
**Learning:** `model.predict()` in TensorFlow/Keras adds massive overhead due to its batch processing setup, making it unsuitable for real-time single-item inference.
**Action:** Always use direct model invocation `model(inputs, training=False).numpy()` for latency-sensitive, single-item prediction tasks.
## 2024-06-12 - [Numpy Operations Overhead on Lists]
**Learning:** Performing numpy operations like `np.argmax` or `np.max` on python lists is significantly slower (~2x) than performing them on native numpy arrays. The implicit conversion to numpy arrays under the hood adds unnecessary overhead.
**Action:** Always perform numpy operations directly on numpy arrays before converting them to python lists for things like JSON serialization.
