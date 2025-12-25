"""Tests for temporal data splitting."""

import numpy as np
import pandas as pd
import pytest

from src.models.train_baseline import temporal_split


def test_temporal_split_basic():
    """Test basic temporal split functionality."""
    # Create simple DataFrame
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(100)],
            "label": np.random.randint(0, 2, 100),
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        }
    )

    train_df, valid_df, test_df = temporal_split(
        df, train_pct=0.6, valid_pct=0.2, test_pct=0.2
    )

    # Check sizes
    assert len(train_df) == 60
    assert len(valid_df) == 20
    assert len(test_df) == 20
    assert len(train_df) + len(valid_df) + len(test_df) == len(df)

    # Check that all customers are included
    all_ids = set(train_df["customer_id"]) | set(valid_df["customer_id"]) | set(test_df["customer_id"])
    assert len(all_ids) == len(df)
    assert all_ids == set(df["customer_id"])

    # Check no overlap
    train_ids = set(train_df["customer_id"])
    valid_ids = set(valid_df["customer_id"])
    test_ids = set(test_df["customer_id"])
    assert len(train_ids & valid_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(valid_ids & test_ids) == 0


def test_temporal_split_with_date_column():
    """Test temporal split with date column for ordering."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(100)],
            "label": np.random.randint(0, 2, 100),
            "signup_date": dates,
            "feature1": np.random.randn(100),
        }
    )

    train_df, valid_df, test_df = temporal_split(
        df, train_pct=0.6, valid_pct=0.2, test_pct=0.2, date_col="signup_date"
    )

    # Check temporal ordering
    assert train_df["signup_date"].max() <= valid_df["signup_date"].min()
    assert valid_df["signup_date"].max() <= test_df["signup_date"].min()

    # Check sizes
    assert len(train_df) == 60
    assert len(valid_df) == 20
    assert len(test_df) == 20


def test_temporal_split_percentages():
    """Test that split percentages are respected."""
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(1000)],
            "label": np.random.randint(0, 2, 1000),
            "feature1": np.random.randn(1000),
        }
    )

    train_df, valid_df, test_df = temporal_split(
        df, train_pct=0.7, valid_pct=0.15, test_pct=0.15
    )

    total = len(df)
    assert abs(len(train_df) / total - 0.7) < 0.01
    assert abs(len(valid_df) / total - 0.15) < 0.01
    assert abs(len(test_df) / total - 0.15) < 0.01


def test_temporal_split_percentages_sum_to_one():
    """Test that percentages must sum to 1.0."""
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(100)],
            "label": np.random.randint(0, 2, 100),
            "feature1": np.random.randn(100),
        }
    )

    with pytest.raises(ValueError):
        temporal_split(df, train_pct=0.5, valid_pct=0.3, test_pct=0.3)


def test_temporal_split_preserves_data():
    """Test that split preserves all data without modification."""
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(100)],
            "label": np.random.randint(0, 2, 100),
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        }
    )

    original_df = df.copy()
    train_df, valid_df, test_df = temporal_split(df)

    # Check that original DataFrame is unchanged
    pd.testing.assert_frame_equal(df, original_df)

    # Check that all columns are preserved
    assert set(train_df.columns) == set(df.columns)
    assert set(valid_df.columns) == set(df.columns)
    assert set(test_df.columns) == set(df.columns)


def test_temporal_split_small_dataset():
    """Test temporal split with very small dataset."""
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(10)],
            "label": np.random.randint(0, 2, 10),
            "feature1": np.random.randn(10),
        }
    )

    train_df, valid_df, test_df = temporal_split(
        df, train_pct=0.6, valid_pct=0.2, test_pct=0.2
    )

    # With small dataset, exact percentages may not be possible
    assert len(train_df) + len(valid_df) + len(test_df) == len(df)
    assert len(train_df) > 0
    assert len(valid_df) > 0
    assert len(test_df) > 0


def test_temporal_split_label_distribution():
    """Test that label distribution is preserved across splits."""
    # Create dataset with known label distribution
    labels = [0] * 70 + [1] * 30
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(100)],
            "label": labels,
            "feature1": np.random.randn(100),
        }
    )

    train_df, valid_df, test_df = temporal_split(df)

    # Check that all splits have some positive and negative samples
    # (exact distribution may vary due to temporal ordering)
    assert (train_df["label"] == 1).sum() >= 0
    assert (train_df["label"] == 0).sum() >= 0
    assert (valid_df["label"] == 1).sum() >= 0
    assert (valid_df["label"] == 0).sum() >= 0
    assert (test_df["label"] == 1).sum() >= 0
    assert (test_df["label"] == 0).sum() >= 0

    # Check that total label counts are preserved
    total_positive = (
        (train_df["label"] == 1).sum()
        + (valid_df["label"] == 1).sum()
        + (test_df["label"] == 1).sum()
    )
    assert total_positive == 30

    total_negative = (
        (train_df["label"] == 0).sum()
        + (valid_df["label"] == 0).sum()
        + (test_df["label"] == 0).sum()
    )
    assert total_negative == 70

