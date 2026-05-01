"""Shared fixtures — physiologically realistic ECoG signals + ARPAbet phonemes."""

import numpy as np
import pytest

ARPABET = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "SIL",
    "SP",
]


def ecog(ch, ts, seed=42):
    """Deterministic ECoG: alpha+beta+high-gamma bands, normalized [0,1]."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1.0, ts, dtype=np.float64)
    sig = np.zeros((ts, ch), dtype=np.float64)
    for c in range(ch):
        ph = rng.uniform(0, 2 * np.pi, 3)
        sig[:, c] = (
            20 * np.sin(2 * np.pi * (8 + c * 0.2) * t + ph[0])
            + 10 * np.sin(2 * np.pi * (18 + c * 0.3) * t + ph[1])
            + 5 * np.sin(2 * np.pi * (80 + c * 1.5) * t + ph[2])
            + rng.randn(ts) * 2
        )
    mn, mx = sig.min(), sig.max()
    return ((sig - mn) / (mx - mn) if mx > mn else np.zeros_like(sig)).astype(
        np.float32
    )


def labels(ts, cls, seed=42):
    """Deterministic one-hot phoneme labels."""
    oh = np.zeros((ts, cls), dtype=np.float32)
    bs = max(1, ts // 5)
    for i in range(ts):
        oh[i, (i // bs) % cls] = 1.0
    return oh


@pytest.fixture
def ecog_2x5x8():
    return np.stack([ecog(8, 5, seed=300 + i) for i in range(2)])


@pytest.fixture
def ecog_1x5x8():
    return ecog(8, 5, seed=200).reshape(1, 5, 8)


@pytest.fixture
def ecog_4x5x8():
    return np.stack([ecog(8, 5, seed=100 + i) for i in range(4)])


@pytest.fixture
def labels_4x5x3():
    return np.stack([labels(5, 3, seed=400 + i) for i in range(4)])


@pytest.fixture
def frame16():
    return ecog(16, 1, seed=500).flatten()


@pytest.fixture
def frame8():
    return ecog(8, 1, seed=600).flatten()


@pytest.fixture
def rt_frame16():
    return ecog(16, 1, seed=700).reshape(1, 1, 16)
