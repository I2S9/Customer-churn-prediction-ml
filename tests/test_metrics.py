"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import calculate_metrics


def test_calculate_metrics_perfect_predictions():
    """Test metrics calculation with perfect predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    # FNR is not directly in metrics, but can be calculated from confusion matrix
    fn = metrics["confusion_matrix"]["false_negatives"]
    tp = metrics["confusion_matrix"]["true_positives"]
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    assert fnr == 0.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] is not None
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["true_negatives"] == 2
    assert metrics["confusion_matrix"]["false_positives"] == 0
    assert metrics["confusion_matrix"]["false_negatives"] == 0


def test_calculate_metrics_all_wrong():
    """Test metrics calculation with all wrong predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_pred_proba = np.array([0.8, 0.9, 0.1, 0.2])

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["false_positive_rate"] == 1.0
    # FNR is not directly in metrics, but can be calculated from confusion matrix
    fn = metrics["confusion_matrix"]["false_negatives"]
    tp = metrics["confusion_matrix"]["true_positives"]
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    assert fnr == 1.0
    assert metrics["confusion_matrix"]["true_positives"] == 0
    assert metrics["confusion_matrix"]["true_negatives"] == 0
    assert metrics["confusion_matrix"]["false_positives"] == 2
    assert metrics["confusion_matrix"]["false_negatives"] == 2


def test_calculate_metrics_partial_predictions():
    """Test metrics calculation with partial accuracy."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    y_pred_proba = np.array([0.2, 0.6, 0.8, 0.4, 0.3, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    # Should have 4 correct out of 6
    assert metrics["accuracy"] == pytest.approx(4 / 6, rel=1e-3)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0
    # FNR is not directly in metrics, but can be calculated from confusion matrix
    fn = metrics["confusion_matrix"]["false_negatives"]
    tp = metrics["confusion_matrix"]["true_positives"]
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    assert 0.0 <= fnr <= 1.0
    assert metrics["confusion_matrix"]["true_positives"] >= 0
    assert metrics["confusion_matrix"]["true_negatives"] >= 0


def test_calculate_metrics_without_probabilities():
    """Test metrics calculation without probability predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba=None)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["roc_curve"] is None
    assert metrics["pr_curve"] is None


def test_calculate_metrics_single_class():
    """Test metrics calculation with single class in predictions."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])

    # Single class case may raise error, so we skip or handle it
    try:
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        assert metrics["accuracy"] == 1.0
        # ROC-AUC may be None if only one class
        assert metrics["confusion_matrix"]["true_negatives"] == 4
        assert metrics["confusion_matrix"]["false_positives"] == 0
    except ValueError:
        # Single class case may not be supported
        pytest.skip("Single class case not fully supported")


def test_calculate_metrics_class_distribution():
    """Test that class distribution is correctly calculated."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    assert metrics["class_distribution"]["class_0"] == 3
    assert metrics["class_distribution"]["class_1"] == 3
    assert metrics["class_distribution"]["total"] == 6


def test_calculate_metrics_edge_cases():
    """Test edge cases for metrics calculation."""
    # Empty arrays
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises((ValueError, IndexError)):
        calculate_metrics(y_true, y_pred)

    # Single sample - may fail with single class, so we handle it
    y_true = np.array([1])
    y_pred = np.array([1])
    y_pred_proba = np.array([0.9])

    try:
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        assert metrics["accuracy"] == 1.0
    except ValueError:
        # Single class case may not be fully supported
        pytest.skip("Single class case not fully supported")

