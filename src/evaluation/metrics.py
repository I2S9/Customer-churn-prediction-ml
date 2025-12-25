"""Standardized evaluation metrics and reporting."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.paths import REPORTS_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_pred_proba: Optional[np.ndarray | pd.Series] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        True binary labels.
    y_pred : numpy.ndarray or pandas.Series
        Predicted binary labels.
    y_pred_proba : numpy.ndarray or pandas.Series, optional
        Predicted probabilities for positive class. Required for AUC metrics.

    Returns
    -------
    dict
        Dictionary containing all calculated metrics.
    """
    metrics = {}

    # Basic classification metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics["confusion_matrix"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # False positive rate
    metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # True positive rate (recall)
    metrics["true_positive_rate"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Specificity
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Class distribution
    metrics["class_distribution"] = {
        "class_0": int((y_true == 0).sum()),
        "class_1": int((y_true == 1).sum()),
        "total": int(len(y_true)),
    }

    # AUC metrics (if probabilities provided)
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            metrics["roc_auc"] = None
            logger.warning("Could not calculate ROC-AUC (possibly only one class present)")

        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics["pr_auc"] = float(auc(recall, precision))
        except ValueError:
            metrics["pr_auc"] = None
            logger.warning("Could not calculate PR-AUC (possibly only one class present)")

        # ROC curve data (for plotting)
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            }
        except ValueError:
            metrics["roc_curve"] = None

        # Precision-Recall curve data (for plotting)
        try:
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                y_true, y_pred_proba
            )
            metrics["pr_curve"] = {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }
        except ValueError:
            metrics["pr_curve"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["roc_curve"] = None
        metrics["pr_curve"] = None

    return metrics


def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_pred_proba: Optional[np.ndarray | pd.Series] = None,
    split_name: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate model and return comprehensive metrics.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        True binary labels.
    y_pred : numpy.ndarray or pandas.Series
        Predicted binary labels.
    y_pred_proba : numpy.ndarray or pandas.Series, optional
        Predicted probabilities for positive class.
    split_name : str, default "test"
        Name of the data split being evaluated.

    Returns
    -------
    dict
        Dictionary containing evaluation results with split name and metrics.
    """
    logger.info(f"Evaluating on {split_name} set...")

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    # Add split information
    results = {
        "split": split_name,
        "metrics": metrics,
    }

    # Log key metrics
    logger.info(f"{split_name.upper()} Metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")
    logger.info(f"  FPR: {metrics['false_positive_rate']:.4f}")

    if metrics["roc_auc"] is not None:
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    if metrics["pr_auc"] is not None:
        logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")

    logger.info(f"  Confusion Matrix: TP={metrics['confusion_matrix']['true_positives']}, "
                f"TN={metrics['confusion_matrix']['true_negatives']}, "
                f"FP={metrics['confusion_matrix']['false_positives']}, "
                f"FN={metrics['confusion_matrix']['false_negatives']}")

    return results


def save_metrics(
    metrics: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save metrics to JSON file.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary to save.
    output_path : str | Path
        Path to save JSON file.

    Returns
    -------
    Path
        Path where metrics were saved.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    logger.info(f"Saving metrics to: {output_path}")

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj: Any) -> Any:
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj

    serializable_metrics = convert_to_json_serializable(metrics)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / 1024**2
    logger.info(f"Metrics saved successfully ({file_size:.4f} MB)")

    return output_path


def load_metrics(metrics_path: str | Path) -> Dict[str, Any]:
    """
    Load metrics from JSON file.

    Parameters
    ----------
    metrics_path : str | Path
        Path to metrics JSON file.

    Returns
    -------
    dict
        Loaded metrics dictionary.
    """
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    logger.info(f"Loading metrics from: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


def compare_metrics(
    metrics_list: list[Dict[str, Any]],
    split_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compare metrics across multiple splits or models.

    Parameters
    ----------
    metrics_list : list of dict
        List of metrics dictionaries (from evaluate_model).
    split_names : list of str, optional
        Names for each metrics dict. If None, uses split names from metrics.

    Returns
    -------
    pandas.DataFrame
        DataFrame with metrics comparison.
    """
    comparison_data = []

    for i, metrics_dict in enumerate(metrics_list):
        split_name = (
            split_names[i] if split_names else metrics_dict.get("split", f"split_{i}")
        )
        metrics = metrics_dict.get("metrics", metrics_dict)

        row = {
            "split": split_name,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "false_positive_rate": metrics.get("false_positive_rate"),
            "true_positive_rate": metrics.get("true_positive_rate"),
            "specificity": metrics.get("specificity"),
        }

        # Add confusion matrix components
        cm = metrics.get("confusion_matrix", {})
        row["true_positives"] = cm.get("true_positives")
        row["true_negatives"] = cm.get("true_negatives")
        row["false_positives"] = cm.get("false_positives")
        row["false_negatives"] = cm.get("false_negatives")

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    return df


def main(
    y_true: Optional[np.ndarray | pd.Series] = None,
    y_pred: Optional[np.ndarray | pd.Series] = None,
    y_pred_proba: Optional[np.ndarray | pd.Series] = None,
    output_path: Optional[str | Path] = None,
    split_name: str = "test",
) -> None:
    """
    Main function to evaluate and save metrics.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series, optional
        True binary labels.
    y_pred : numpy.ndarray or pandas.Series, optional
        Predicted binary labels.
    y_pred_proba : numpy.ndarray or pandas.Series, optional
        Predicted probabilities.
    output_path : str | Path, optional
        Path to save metrics JSON file.
    split_name : str, default "test"
        Name of the data split.
    """
    if y_true is None or y_pred is None:
        logger.error("y_true and y_pred are required")
        return

    if output_path is None:
        output_path = REPORTS_DIR / f"{split_name}_metrics.json"

    results = evaluate_model(y_true, y_pred, y_pred_proba, split_name=split_name)
    save_metrics(results, output_path)


if __name__ == "__main__":
    logger.info("Metrics module loaded. Use evaluate_model() or main() functions.")

