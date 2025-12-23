"""Standardized evaluation framework for churn prediction models.

This module provides comprehensive metrics calculation and reporting:
- AUC, precision, recall, F1, FPR
- Confusion matrix
- JSON report generation
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def calculate_metrics(y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class (optional)
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # False Positive Rate
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["tnr"] = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0  # True Negative Rate
    else:
        # Handle edge case with single class
        metrics["fpr"] = 0.0
        metrics["tnr"] = 1.0
    
    # ROC-AUC (requires probabilities)
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None
    
    # Confusion matrix
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]) if len(cm) > 0 else 0,
        "fp": int(cm[0, 1]) if len(cm) > 1 else 0,
        "fn": int(cm[1, 0]) if len(cm) > 1 else 0,
        "tp": int(cm[1, 1]) if len(cm) > 1 else 0,
    }
    
    # Class distribution
    metrics["class_distribution"] = {
        "negative": int((y_true == 0).sum()),
        "positive": int((y_true == 1).sum()),
        "total": int(len(y_true)),
    }
    
    # Additional metrics
    metrics["support"] = {
        "negative": int((y_true == 0).sum()),
        "positive": int((y_true == 1).sum()),
    }
    
    return metrics


def calculate_curve_metrics(y_true, y_proba) -> Dict[str, Any]:
    """Calculate ROC and Precision-Recall curve metrics.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
    
    Returns:
        Dictionary containing curve data
    """
    curves = {}
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    curves["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
    }
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    curves["pr_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    }
    
    return curves


def generate_classification_report(y_true, y_pred, target_names=None) -> str:
    """Generate sklearn classification report as string.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        target_names: Optional names for classes
    
    Returns:
        Classification report string
    """
    if target_names is None:
        target_names = ["No Churn", "Churn"]
    
    return classification_report(y_true, y_pred, target_names=target_names)


def evaluate_model(y_true, y_pred, y_proba: Optional[np.ndarray] = None,
                 include_curves: bool = False) -> Dict[str, Any]:
    """Comprehensive model evaluation.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class (optional)
        include_curves: Whether to include ROC/PR curve data
    
    Returns:
        Complete evaluation report dictionary
    """
    report = {
        "metrics": calculate_metrics(y_true, y_pred, y_proba),
    }
    
    # Add curve data if requested and probabilities available
    if include_curves and y_proba is not None:
        report["curves"] = calculate_curve_metrics(y_true, y_proba)
    
    # Add classification report
    report["classification_report"] = generate_classification_report(y_true, y_pred)
    
    return report


def save_evaluation_report(report: Dict[str, Any], filepath: Path,
                          model_name: str = "model",
                          dataset_name: str = "dataset",
                          additional_metadata: Optional[Dict] = None):
    """Save evaluation report to JSON file.
    
    Args:
        report: Evaluation report dictionary
        filepath: Path to save JSON file
        model_name: Name of the model
        dataset_name: Name of the dataset/split
        additional_metadata: Optional additional metadata to include
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create complete report with metadata
    complete_report = {
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "evaluation_framework_version": "1.0",
        },
        "evaluation": report,
    }
    
    # Add additional metadata if provided
    if additional_metadata:
        complete_report["metadata"].update(additional_metadata)
    
    # Save to JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(complete_report, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation report saved to: {filepath}")


def print_metrics_summary(metrics: Dict[str, Any], set_name: str = ""):
    """Print a formatted summary of metrics.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        set_name: Name of the dataset (e.g., "validation", "test")
    """
    prefix = f"{set_name}: " if set_name else ""
    
    print(f"\n{prefix}Metrics Summary")
    print("-" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}")
    
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  True Negatives:  {cm['tn']}")
    print(f"  False Positives: {cm['fp']}")
    print(f"  False Negatives: {cm['fn']}")
    print(f"  True Positives:  {cm['tp']}")
    
    print(f"\nClass Distribution:")
    dist = metrics["class_distribution"]
    print(f"  Negative: {dist['negative']} ({dist['negative']/dist['total']*100:.1f}%)")
    print(f"  Positive: {dist['positive']} ({dist['positive']/dist['total']*100:.1f}%)")


def load_evaluation_report(filepath: Path) -> Dict[str, Any]:
    """Load evaluation report from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Evaluation report dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_models(reports: List[Dict[str, Any]], model_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare multiple model evaluation reports.
    
    Args:
        reports: List of evaluation report dictionaries
        model_names: Optional list of model names
    
    Returns:
        DataFrame with comparison metrics
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(reports))]
    
    comparison_data = []
    
    for report, name in zip(reports, model_names):
        metrics = report.get("metrics", {})
        comparison_data.append({
            "model": name,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "fpr": metrics.get("fpr"),
        })
    
    return pd.DataFrame(comparison_data)

