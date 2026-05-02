# The James Bridge 
## (Proprietary BCI Architecture)
**A high-performance Brain-Computer Interface engineered for real-time neurorehabilitation.**

---

# Mission: The Why
The James Bridge is named in honor of my cousin, **James Shelton**. From age 13 to 24, I served as James’s sole caretaker. This project is a proprietary, independent effort to bridge the gap between neural intent and functional independence, built with enterprise software rigor from Day 1.

## System Architecture & Performance
Designed for **latency-critical** medical applications. The system achieves a **sub-100ms full-pipeline processing latency**, powered by a highly compressed **0.17ms isolated model inference speed**.

### **Neural Processing Pipeline:**
* **Acquisition:** Edge-hardware integration (ESP32/OpenBCI) synchronized via LSL (Lab Streaming Layer) for 8-16 channel EEG/ECoG.
* **Preprocessing:** Real-time Bandpass (0.5-50Hz), Notch (60Hz), and Common Average Referencing (CAR) with sub-millisecond artifact rejection.
* **Feature Extraction:** Spatio-temporal feature mapping using a Hybrid 1D-CNN-LSTM architecture.
* **Inference Engine:** High-concurrency local inference utilizing direct tensor invocation to bypass standard framework overhead.

#### **Current Tech Stack (V1):**
* **Language:** Python 3.11+
* **Deep Learning:** TensorFlow / Keras (Core Architecture)
* **Signal Processing:** MNE-Python, NumPy, SciPy, Numba (JIT compilation)
* **Infrastructure:** Docker, GCP, Git CI/CD

##### Key Engineering Feats
* **Streaming Enrollment:** Custom orchestration (`production_train.py`) that syncs massive clinical datasets (ds003620) from AWS S3, trains the weights, and immediately purges raw subject data to bypass tight storage constraints.
* **Artifact Suppression:** Real-time ICA and Numba-optimized filters for EOG/EMG ocular artifact removal.
* **Clinical-Grade Reliability:** 100% actuation test coverage ensuring deterministic motor-command dispatch.

## 🗺️ Roadmap (Upcoming Sprints)
* **Phase 4 (Edge Compute Migration):** Implementation of **ONNX Runtime** and TensorFlow Lite Micro for bare-metal C++ microcontroller deployment.
* **Phase 5 (Agentic Integration):** Integration of an Agent Development Kit (ADK) to autonomously summarize real-time neural trends for clinical and medical review.

---
*Developed & Maintained by Donavan Eugene Pyle, II*
