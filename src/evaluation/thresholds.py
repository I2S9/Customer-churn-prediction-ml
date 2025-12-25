"""Threshold analysis for decision-making under business constraints."""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from src.utils.paths import REPORTS_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_cost(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> float:
    """
    Calculate total cost based on false positives and false negatives.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        True binary labels.
    y_pred : numpy.ndarray or pandas.Series
        Predicted binary labels.
    cost_fp : float, default 1.0
        Cost per false positive (e.g., cost of unnecessary retention campaign).
    cost_fn : float, default 1.0
        Cost per false negative (e.g., cost of lost customer).

    Returns
    -------
    float
        Total cost: cost_fp * FP + cost_fn * FN
    """
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    total_cost = cost_fp * fp + cost_fn * fn
    return float(total_cost)


def analyze_thresholds(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Analyze metrics and costs across different classification thresholds.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        True binary labels.
    y_pred_proba : numpy.ndarray or pandas.Series
        Predicted probabilities for positive class.
    cost_fp : float, default 1.0
        Cost per false positive.
    cost_fn : float, default 5.0
        Cost per false negative.
    thresholds : numpy.ndarray, optional
        Thresholds to evaluate. If None, uses 100 evenly spaced thresholds.

    Returns
    -------
    pandas.DataFrame
        DataFrame with metrics and costs for each threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 100)

    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        cost = calculate_cost(y_true, y_pred, cost_fp=cost_fp, cost_fn=cost_fn)

        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0,
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "total_cost": float(cost),
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
            }
        )

    df = pd.DataFrame(results)
    return df


def find_optimal_threshold(
    threshold_df: pd.DataFrame,
    metric: str = "total_cost",
    minimize: bool = True,
) -> float:
    """
    Find optimal threshold based on a metric.

    Parameters
    ----------
    threshold_df : pandas.DataFrame
        DataFrame from analyze_thresholds.
    metric : str, default "total_cost"
        Metric to optimize.
    minimize : bool, default True
        If True, minimize the metric. If False, maximize it.

    Returns
    -------
    float
        Optimal threshold value.
    """
    if minimize:
        optimal_idx = threshold_df[metric].idxmin()
    else:
        optimal_idx = threshold_df[metric].idxmax()

    optimal_threshold = threshold_df.loc[optimal_idx, "threshold"]
    optimal_value = threshold_df.loc[optimal_idx, metric]

    logger.info(f"Optimal threshold: {optimal_threshold:.4f} ({metric}={optimal_value:.4f})")

    return float(optimal_threshold)


def top_k_targeting(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    k_values: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Analyze top-k targeting strategy.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        True binary labels.
    y_pred_proba : numpy.ndarray or pandas.Series
        Predicted probabilities for positive class.
    k_values : list of int, optional
        List of k values to analyze. If None, uses [10, 20, 30, 40, 50].

    Returns
    -------
    pandas.DataFrame
        DataFrame with metrics for each k value.
    """
    if k_values is None:
        k_values = [10, 20, 30, 40, 50]

    results = []

    n_total = len(y_true)
    sorted_indices = np.argsort(y_pred_proba)[::-1]  # Sort descending

    for k in k_values:
        k_pct = k / 100.0
        k_absolute = int(n_total * k_pct)

        # Top k predictions
        top_k_indices = sorted_indices[:k_absolute]
        y_pred_top_k = np.zeros(n_total, dtype=int)
        y_pred_top_k[top_k_indices] = 1

        tp = ((y_pred_top_k == 1) & (y_true == 1)).sum()
        fp = ((y_pred_top_k == 1) & (y_true == 0)).sum()
        fn = ((y_true == 1) & (y_pred_top_k == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Churn rate in top k
        churn_rate_top_k = y_true[top_k_indices].mean() if len(top_k_indices) > 0 else 0.0

        results.append(
            {
                "k_percent": k,
                "k_absolute": k_absolute,
                "precision": float(precision),
                "recall": float(recall),
                "churn_rate_in_top_k": float(churn_rate_top_k),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        )

    df = pd.DataFrame(results)
    return df


def plot_precision_recall_vs_threshold(
    threshold_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Plot precision and recall vs threshold.

    Parameters
    ----------
    threshold_df : pandas.DataFrame
        DataFrame from analyze_thresholds.
    output_path : str | Path, optional
        Path to save figure.

    Returns
    -------
    Path
        Path where figure was saved.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "precision_recall_vs_threshold.png"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        label="Precision",
        linewidth=2,
    )
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", linewidth=2)
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision and Recall vs Classification Threshold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved precision-recall plot to: {output_path}")
    return output_path


def plot_cost_vs_threshold(
    threshold_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Plot total cost vs threshold.

    Parameters
    ----------
    threshold_df : pandas.DataFrame
        DataFrame from analyze_thresholds.
    output_path : str | Path, optional
        Path to save figure.

    Returns
    -------
    Path
        Path where figure was saved.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "cost_vs_threshold.png"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        threshold_df["threshold"],
        threshold_df["total_cost"],
        label="Total Cost",
        linewidth=2,
        color="red",
    )

    # Mark optimal threshold
    optimal_idx = threshold_df["total_cost"].idxmin()
    optimal_threshold = threshold_df.loc[optimal_idx, "threshold"]
    optimal_cost = threshold_df.loc[optimal_idx, "total_cost"]
    ax.axvline(optimal_threshold, color="green", linestyle="--", label="Optimal Threshold")
    ax.plot(optimal_threshold, optimal_cost, "go", markersize=10)

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Total Cost", fontsize=12)
    ax.set_title(
        f"Total Cost vs Threshold (FP cost={threshold_df['cost_fp'].iloc[0]:.1f}, "
        f"FN cost={threshold_df['cost_fn'].iloc[0]:.1f})",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved cost plot to: {output_path}")
    return output_path


def plot_top_k_analysis(
    top_k_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Plot top-k targeting analysis.

    Parameters
    ----------
    top_k_df : pandas.DataFrame
        DataFrame from top_k_targeting.
    output_path : str | Path, optional
        Path to save figure.

    Returns
    -------
    Path
        Path where figure was saved.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "top_k_targeting.png"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Precision and Recall vs k
    ax1.plot(
        top_k_df["k_percent"],
        top_k_df["precision"],
        label="Precision",
        marker="o",
        linewidth=2,
    )
    ax1.plot(
        top_k_df["k_percent"],
        top_k_df["recall"],
        label="Recall",
        marker="s",
        linewidth=2,
    )
    ax1.set_xlabel("Top k (%)", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Precision and Recall vs Top-k", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Churn rate in top k
    ax2.plot(
        top_k_df["k_percent"],
        top_k_df["churn_rate_in_top_k"],
        label="Churn Rate in Top-k",
        marker="o",
        linewidth=2,
        color="purple",
    )
    ax2.axhline(
        top_k_df["churn_rate_in_top_k"].iloc[-1] if len(top_k_df) > 0 else 0,
        color="red",
        linestyle="--",
        label="Overall Churn Rate",
    )
    ax2.set_xlabel("Top k (%)", fontsize=12)
    ax2.set_ylabel("Churn Rate", fontsize=12)
    ax2.set_title("Churn Rate in Top-k vs Overall", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved top-k analysis plot to: {output_path}")
    return output_path


def perform_threshold_analysis(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """
    Perform comprehensive threshold analysis.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        True binary labels.
    y_pred_proba : numpy.ndarray or pandas.Series
        Predicted probabilities for positive class.
    cost_fp : float, default 1.0
        Cost per false positive.
    cost_fn : float, default 5.0
        Cost per false negative.
    output_dir : str | Path, optional
        Directory to save outputs. Defaults to reports/.

    Returns
    -------
    dict
        Dictionary containing all analysis results.
    """
    if output_dir is None:
        output_dir = REPORTS_DIR
    else:
        output_dir = Path(output_dir)

    ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("Threshold Analysis for Decision-Making")
    logger.info("=" * 60)
    logger.info(f"Cost FP: {cost_fp}, Cost FN: {cost_fn}")

    # Threshold analysis
    logger.info("Analyzing thresholds...")
    threshold_df = analyze_thresholds(y_true, y_pred_proba, cost_fp=cost_fp, cost_fn=cost_fn)
    optimal_threshold = find_optimal_threshold(threshold_df, metric="total_cost", minimize=True)

    # Top-k analysis
    logger.info("Analyzing top-k targeting...")
    top_k_df = top_k_targeting(y_true, y_pred_proba)

    # Generate plots
    logger.info("Generating plots...")
    plot_precision_recall_vs_threshold(threshold_df, output_dir / "precision_recall_vs_threshold.png")
    plot_cost_vs_threshold(threshold_df, output_dir / "cost_vs_threshold.png")
    plot_top_k_analysis(top_k_df, output_dir / "top_k_targeting.png")

    # Compile results
    results = {
        "cost_config": {
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
        },
        "optimal_threshold": float(optimal_threshold),
        "optimal_threshold_metrics": threshold_df[
            threshold_df["threshold"] == optimal_threshold
        ].iloc[0].to_dict(),
        "threshold_analysis": threshold_df.to_dict("records"),
        "top_k_analysis": top_k_df.to_dict("records"),
    }

    # Save results
    output_path = output_dir / "threshold_analysis.json"
    logger.info(f"Saving results to: {output_path}")

    def convert_to_json_serializable(obj):
        """Convert numpy types to native Python types."""
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

    serializable_results = convert_to_json_serializable(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Threshold analysis completed successfully")
    logger.info("=" * 60)

    return results


def main(
    y_true_path: Optional[str] = None,
    y_pred_proba_path: Optional[str] = None,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    output_dir: Optional[str] = None,
) -> None:
    """
    Main function to perform threshold analysis.

    Parameters
    ----------
    y_true_path : str, optional
        Path to CSV/Parquet file with true labels (column 'label').
    y_pred_proba_path : str, optional
        Path to CSV/Parquet file with predicted probabilities (column 'proba').
    cost_fp : float, default 1.0
        Cost per false positive.
    cost_fn : float, default 5.0
        Cost per false negative.
    output_dir : str, optional
        Directory to save outputs.
    """
    logger.info("Threshold analysis module loaded. Use perform_threshold_analysis() function.")


if __name__ == "__main__":
    logger.info("Threshold analysis module loaded. Use perform_threshold_analysis() function.")

