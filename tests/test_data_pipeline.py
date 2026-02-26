import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from neurobridge.config import DatasetConfig
from neurobridge.data_pipeline import ECoGDatasetBuilder, load_phoneme_intervals, PhonemeInventory

@pytest.fixture
def mock_dataset_config(tmp_path):
    config = DatasetConfig(
        ecog_dir=tmp_path / "ecog",
        labels_dir=tmp_path / "labels",
        file_extension=".mat",
        label_extension=".csv"
    )
    config.ecog_dir.mkdir(parents=True)
    config.labels_dir.mkdir(parents=True)
    return config

@pytest.fixture
def mock_inventory():
    return PhonemeInventory(symbols=["a", "b", "c"])

def test_select_split_disjoint(mock_dataset_config, mock_inventory):
    # Create 10 mock file pairs
    pairs = []
    for i in range(10):
        ecog = mock_dataset_config.ecog_dir / f"subj_{i}.mat"
        ecog.touch()
        label = mock_dataset_config.labels_dir / f"subj_{i}.csv"
        label.touch()
        pairs.append((ecog, label))

    # Initialize builder (will scan, but we override pairs)
    builder = ECoGDatasetBuilder(mock_dataset_config, mock_inventory, split="train", shuffle_files=False)
    # Manually inject pairs to avoid _scan_pairs logic and ensure order matches our test expectation
    # (though shuffle_files=False in init helps, _scan_pairs relies on file system order which is usually sorted)

    # Test Train
    builder.split = "train"
    train_pairs = builder._select_split(pairs)

    # Test Val
    builder.split = "val"
    val_pairs = builder._select_split(pairs)

    # Test Test
    builder.split = "test"
    test_pairs = builder._select_split(pairs)

    # Verify counts (default 0.1 val, 0.1 test -> 8 train, 1 val, 1 test)
    assert len(train_pairs) == 8
    assert len(val_pairs) == 1
    assert len(test_pairs) == 1

    # Verify disjointness
    train_set = set(train_pairs)
    val_set = set(val_pairs)
    test_set = set(test_pairs)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

def test_select_split_small_dataset(mock_dataset_config, mock_inventory):
    # Create 1 mock file pair
    ecog = mock_dataset_config.ecog_dir / "subj_0.mat"
    ecog.touch()
    label = mock_dataset_config.labels_dir / "subj_0.csv"
    label.touch()
    pairs = [(ecog, label)]

    builder = ECoGDatasetBuilder(mock_dataset_config, mock_inventory, split="train", shuffle_files=False)

    # Test Train
    builder.split = "train"
    train_pairs = builder._select_split(pairs)
    assert len(train_pairs) == 1

    # Test Val (should be empty, NOT overlapping)
    builder.split = "val"
    val_pairs = builder._select_split(pairs)
    assert len(val_pairs) == 0

    # Test Test (should be empty, NOT overlapping)
    builder.split = "test"
    test_pairs = builder._select_split(pairs)
    assert len(test_pairs) == 0

def test_load_phoneme_intervals_missing_columns(mock_dataset_config):
    label_file = mock_dataset_config.labels_dir / "bad.csv"
    # Create CSV with missing 'end_sec'
    pd.DataFrame({"start_sec": [0.0], "phoneme": ["sil"]}).to_csv(label_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_phoneme_intervals(label_file, mock_dataset_config)
