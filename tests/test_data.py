"""
Tests for NeuroBridge data loading and generation.
"""

import numpy as np
import pytest


class TestMockDataGeneration:
    """Test mock data generators."""

    def test_mock_ecog_shape(self):
        """Mock ECoG data should have correct shape."""
        from neurobridge.data import generate_mock_ecog_data

        data = generate_mock_ecog_data(5, timesteps=10, features=16)
        assert data.shape == (5, 10, 16)

    def test_mock_ecog_dtype(self):
        """Mock ECoG data should be float32."""
        from neurobridge.data import generate_mock_ecog_data

        data = generate_mock_ecog_data(3, timesteps=5, features=8)
        assert data.dtype == np.float32

    def test_mock_ecog_values_in_range(self):
        """Mock ECoG data should be in [0, 1] (random uniform)."""
        from neurobridge.data import generate_mock_ecog_data

        data = generate_mock_ecog_data(10, timesteps=5, features=8)
        assert data.min() >= 0.0
        assert data.max() <= 1.0

    def test_mock_phoneme_labels_shape(self):
        """Mock phoneme labels should have correct shape and be one-hot."""
        from neurobridge.data import generate_mock_phoneme_labels

        labels = generate_mock_phoneme_labels(5, timesteps=10, num_classes=3)
        assert labels.shape == (5, 10, 3)

    def test_mock_phoneme_labels_are_onehot(self):
        """Each timestep should have exactly one 1.0 value (one-hot)."""
        from neurobridge.data import generate_mock_phoneme_labels

        labels = generate_mock_phoneme_labels(5, timesteps=10, num_classes=3)
        sums = labels.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0)

    def test_data_generator_yields_batches(self):
        """Data generator should yield correctly shaped batches."""
        from neurobridge.data import create_data_generator

        gen = create_data_generator(batch_size=4)
        ecog, labels = next(gen)

        assert ecog.shape[0] == 4
        assert labels.shape[0] == 4
        assert ecog.ndim == 3
        assert labels.ndim == 3
