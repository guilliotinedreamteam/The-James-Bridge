import sys
import os
import time

# Ensure project root is on sys.path so this script works when invoked
# directly (e.g., `python tests/benchmark_latency.py`) without PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from neurobridge.processing.artifacts import ArtifactRejector

def test_performance():
    rejector = ArtifactRejector(sfreq=1000)
    data = np.random.randn(64, 1000)
    
    # Warm up Numba
    _ = rejector.full_clinical_clean(data)
    
    # Measure
    start = time.perf_counter()
    for _ in range(10):
        _ = rejector.full_clinical_clean(data)
    avg_latency = (time.perf_counter() - start) / 10
    
    print(f"Average Processing Latency: {avg_latency*1000:.2f}ms")
    assert avg_latency < 0.1, f"Latency exceeded 100ms threshold: {avg_latency*1000:.2f}ms"

if __name__ == "__main__":
    test_performance()
