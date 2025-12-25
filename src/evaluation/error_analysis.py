"""Error analysis and segmentation to identify model failure modes."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.utils.paths import REPORTS_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer segments based on activity, tenure, and usage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features and predictions.

    Returns
    -------
    pandas.DataFrame
        DataFrame with segment columns added.
    """
    df_segmented = df.copy()

    # Segment by activity level
    if "total_events" in df.columns:
        activity_median = df["total_events"].median()
        df_segmented["activity_segment"] = df["total_events"].apply(
            lambda x: "high" if x >= activity_median else "low"
        )
    elif "events_last_30d" in df.columns:
        activity_median = df["events_last_30d"].median()
        df_segmented["activity_segment"] = df["events_last_30d"].apply(
            lambda x: "high" if x >= activity_median else "low"
        )
    else:
        df_segmented["activity_segment"] = "unknown"

    # Segment by tenure (customer age)
    if "customer_age_days" in df.columns:
        tenure_median = df["customer_age_days"].median()
        df_segmented["tenure_segment"] = df["customer_age_days"].apply(
            lambda x: "new" if x < tenure_median else "established"
        )
    elif "days_since_signup" in df.columns:
        tenure_median = df["days_since_signup"].median()
        df_segmented["tenure_segment"] = df["days_since_signup"].apply(
            lambda x: "new" if x < tenure_median else "established"
        )
    else:
        df_segmented["tenure_segment"] = "unknown"

    # Segment by usage type (based on event types or revenue)
    if "count_purchase" in df.columns and "count_login" in df.columns:
        df_segmented["usage_segment"] = df.apply(
            lambda row: "purchaser"
            if row["count_purchase"] > 0
            else "browser"
            if row["count_login"] > 0
            else "inactive",
            axis=1,
        )
    elif "total_revenue" in df.columns:
        revenue_median = df["total_revenue"].median()
        df_segmented["usage_segment"] = df["total_revenue"].apply(
            lambda x: "high_value" if x >= revenue_median else "low_value"
        )
    else:
        df_segmented["usage_segment"] = "unknown"

    return df_segmented


def analyze_errors_by_segment(
    df: pd.DataFrame,
    y_true_col: str = "label",
    y_pred_col: str = "predicted",
    segment_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Analyze error rates by segment.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with true labels, predictions, and segments.
    y_true_col : str, default "label"
        Column name for true labels.
    y_pred_col : str, default "predicted"
        Column name for predictions.
    segment_cols : list of str, optional
        Columns to segment by. If None, uses activity_segment, tenure_segment, usage_segment.

    Returns
    -------
    pandas.DataFrame
        DataFrame with error metrics by segment.
    """
    if segment_cols is None:
        segment_cols = ["activity_segment", "tenure_segment", "usage_segment"]

    # Filter to only existing segment columns
    segment_cols = [col for col in segment_cols if col in df.columns]

    if not segment_cols:
        logger.warning("No segment columns found. Returning empty analysis.")
        return pd.DataFrame()

    results = []

    # Overall metrics
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    overall_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    overall_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    results.append(
        {
            "segment_type": "overall",
            "segment_value": "all",
            "n_samples": len(df),
            "n_churned": int((y_true == 1).sum()),
            "churn_rate": float(y_true.mean()),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "false_negative_rate": float(overall_fnr),
            "false_positive_rate": float(overall_fpr),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        }
    )

    # Analyze by each segment column
    for segment_col in segment_cols:
        for segment_value in df[segment_col].unique():
            segment_df = df[df[segment_col] == segment_value]
            if len(segment_df) == 0:
                continue

            seg_y_true = segment_df[y_true_col]
            seg_y_pred = segment_df[y_pred_col]

            seg_cm = confusion_matrix(seg_y_true, seg_y_pred)
            if seg_cm.size == 4:
                seg_tn, seg_fp, seg_fn, seg_tp = seg_cm.ravel()
            else:
                # Handle edge case with single class
                if seg_cm.size == 1:
                    seg_tn, seg_fp, seg_fn, seg_tp = 0, 0, 0, 0
                    if (seg_y_true == 0).all():
                        seg_tn = len(seg_y_true)
                    elif (seg_y_true == 1).all():
                        seg_fn = len(seg_y_true)
                else:
                    continue

            seg_fnr = seg_fn / (seg_fn + seg_tp) if (seg_fn + seg_tp) > 0 else 0.0
            seg_fpr = seg_fp / (seg_fp + seg_tn) if (seg_fp + seg_tn) > 0 else 0.0

            results.append(
                {
                    "segment_type": segment_col,
                    "segment_value": str(segment_value),
                    "n_samples": len(segment_df),
                    "n_churned": int((seg_y_true == 1).sum()),
                    "churn_rate": float(seg_y_true.mean()),
                    "true_positives": int(seg_tp),
                    "true_negatives": int(seg_tn),
                    "false_positives": int(seg_fp),
                    "false_negatives": int(seg_fn),
                    "false_negative_rate": float(seg_fnr),
                    "false_positive_rate": float(seg_fpr),
                    "precision": float(seg_tp / (seg_tp + seg_fp)) if (seg_tp + seg_fp) > 0 else 0.0,
                    "recall": float(seg_tp / (seg_tp + seg_fn)) if (seg_tp + seg_fn) > 0 else 0.0,
                }
            )

    return pd.DataFrame(results)


def identify_failure_segments(
    error_df: pd.DataFrame,
    fnr_threshold: float = 0.3,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Identify segments with high false negative rates.

    Parameters
    ----------
    error_df : pandas.DataFrame
        DataFrame from analyze_errors_by_segment.
    fnr_threshold : float, default 0.3
        FNR threshold to flag as failure segment.
    min_samples : int, default 10
        Minimum samples required in segment.

    Returns
    -------
    pandas.DataFrame
        Segments flagged as failure modes.
    """
    failure_segments = error_df[
        (error_df["false_negative_rate"] >= fnr_threshold)
        & (error_df["n_samples"] >= min_samples)
        & (error_df["segment_type"] != "overall")
    ].copy()

    failure_segments = failure_segments.sort_values(
        "false_negative_rate", ascending=False
    )

    return failure_segments


def plot_error_analysis(
    error_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Create visualization of error analysis by segment.

    Parameters
    ----------
    error_df : pandas.DataFrame
        DataFrame from analyze_errors_by_segment.
    output_path : str | Path, optional
        Path to save figure.

    Returns
    -------
    Path
        Path where figure was saved.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "error_analysis.png"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    # Filter out overall row
    segment_df = error_df[error_df["segment_type"] != "overall"].copy()

    if len(segment_df) == 0:
        logger.warning("No segment data to plot")
        return output_path

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: FNR by segment type
    ax1 = axes[0, 0]
    for segment_type in segment_df["segment_type"].unique():
        seg_data = segment_df[segment_df["segment_type"] == segment_type]
        ax1.bar(
            seg_data["segment_value"],
            seg_data["false_negative_rate"],
            label=segment_type,
            alpha=0.7,
        )
    ax1.set_xlabel("Segment Value", fontsize=11)
    ax1.set_ylabel("False Negative Rate", fontsize=11)
    ax1.set_title("False Negative Rate by Segment", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: FPR by segment type
    ax2 = axes[0, 1]
    for segment_type in segment_df["segment_type"].unique():
        seg_data = segment_df[segment_df["segment_type"] == segment_type]
        ax2.bar(
            seg_data["segment_value"],
            seg_data["false_positive_rate"],
            label=segment_type,
            alpha=0.7,
        )
    ax2.set_xlabel("Segment Value", fontsize=11)
    ax2.set_ylabel("False Positive Rate", fontsize=11)
    ax2.set_title("False Positive Rate by Segment", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    # Plot 3: Churn rate vs FNR scatter
    ax3 = axes[1, 0]
    for segment_type in segment_df["segment_type"].unique():
        seg_data = segment_df[segment_df["segment_type"] == segment_type]
        ax3.scatter(
            seg_data["churn_rate"],
            seg_data["false_negative_rate"],
            label=segment_type,
            s=100,
            alpha=0.6,
        )
    ax3.set_xlabel("Churn Rate", fontsize=11)
    ax3.set_ylabel("False Negative Rate", fontsize=11)
    ax3.set_title("Churn Rate vs False Negative Rate", fontsize=13, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sample size vs FNR
    ax4 = axes[1, 1]
    for segment_type in segment_df["segment_type"].unique():
        seg_data = segment_df[segment_df["segment_type"] == segment_type]
        ax4.scatter(
            seg_data["n_samples"],
            seg_data["false_negative_rate"],
            label=segment_type,
            s=100,
            alpha=0.6,
        )
    ax4.set_xlabel("Sample Size", fontsize=11)
    ax4.set_ylabel("False Negative Rate", fontsize=11)
    ax4.set_title("Sample Size vs False Negative Rate", fontsize=13, fontweight="bold")
    ax4.set_xscale("log")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved error analysis plot to: {output_path}")
    return output_path


def generate_error_report(
    error_df: pd.DataFrame,
    failure_segments: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Generate markdown report of error analysis.

    Parameters
    ----------
    error_df : pandas.DataFrame
        DataFrame from analyze_errors_by_segment.
    failure_segments : pandas.DataFrame
        Segments flagged as failure modes.
    output_path : str | Path, optional
        Path to save markdown report.

    Returns
    -------
    Path
        Path where report was saved.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "error_analysis.md"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    # Get overall metrics
    overall = error_df[error_df["segment_type"] == "overall"].iloc[0]

    report_lines = [
        "# Error Analysis and Segmentation Report",
        "",
        "## Executive Summary",
        "",
        f"- **Total Samples**: {int(overall['n_samples']):,}",
        f"- **Overall Churn Rate**: {overall['churn_rate']:.2%}",
        f"- **Overall False Negative Rate**: {overall['false_negative_rate']:.2%}",
        f"- **Overall False Positive Rate**: {overall['false_positive_rate']:.2%}",
        f"- **Failure Segments Identified**: {len(failure_segments)}",
        "",
        "## Overall Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| True Positives | {int(overall['true_positives']):,} |",
        f"| True Negatives | {int(overall['true_negatives']):,} |",
        f"| False Positives | {int(overall['false_positives']):,} |",
        f"| False Negatives | {int(overall['false_negatives']):,} |",
        f"| Precision | {overall['precision']:.4f} |",
        f"| Recall | {overall['recall']:.4f} |",
        "",
        "## Error Analysis by Segment",
        "",
    ]

    # Add segment analysis
    segment_types = error_df[error_df["segment_type"] != "overall"]["segment_type"].unique()

    for seg_type in segment_types:
        seg_data = error_df[error_df["segment_type"] == seg_type].copy()
        seg_data = seg_data.sort_values("false_negative_rate", ascending=False)

        report_lines.extend([
            f"### {seg_type.replace('_', ' ').title()}",
            "",
            "| Segment | Samples | Churn Rate | FNR | FPR | Precision | Recall |",
            "|---------|---------|------------|-----|-----|-----------|--------|",
        ])

        for _, row in seg_data.iterrows():
            report_lines.append(
                f"| {row['segment_value']} | "
                f"{int(row['n_samples']):,} | "
                f"{row['churn_rate']:.2%} | "
                f"{row['false_negative_rate']:.2%} | "
                f"{row['false_positive_rate']:.2%} | "
                f"{row['precision']:.4f} | "
                f"{row['recall']:.4f} |"
            )

        report_lines.append("")

    # Add failure segments
    if len(failure_segments) > 0:
        report_lines.extend([
            "## Identified Failure Segments",
            "",
            "The following segments show high False Negative Rates (FNR) and may require "
            "model improvement or targeted interventions:",
            "",
            "| Segment Type | Segment Value | Samples | Churn Rate | FNR | FPR |",
            "|--------------|---------------|---------|------------|-----|-----|",
        ])

        for _, row in failure_segments.iterrows():
            report_lines.append(
                f"| {row['segment_type']} | "
                f"{row['segment_value']} | "
                f"{int(row['n_samples']):,} | "
                f"{row['churn_rate']:.2%} | "
                f"{row['false_negative_rate']:.2%} | "
                f"{row['false_positive_rate']:.2%} |"
            )

        report_lines.append("")
        report_lines.extend([
            "### Recommendations",
            "",
            "1. **Model Improvement**: Consider adding features specific to high-FNR segments",
            "2. **Targeted Interventions**: Implement proactive retention for identified segments",
            "3. **Data Collection**: Gather more data for segments with high FNR and low sample size",
            "4. **Threshold Adjustment**: Consider segment-specific thresholds for high-risk segments",
            "",
        ])
    else:
        report_lines.extend([
            "## Identified Failure Segments",
            "",
            "No segments identified with high False Negative Rate.",
            "",
        ])

    report_lines.extend([
        "## Methodology",
        "",
        "- **Segmentation**: Customers segmented by activity level, tenure, and usage type",
        "- **Error Metrics**: False Negative Rate (FNR) and False Positive Rate (FPR) calculated per segment",
        "- **Failure Identification**: Segments with FNR ≥ 30% and minimum 10 samples flagged",
        "",
    ])

    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved error analysis report to: {output_path}")
    return output_path


def perform_error_analysis(
    df: pd.DataFrame,
    y_true_col: str = "label",
    y_pred_col: str = "predicted",
    fnr_threshold: float = 0.3,
    output_dir: Optional[str | Path] = None,
) -> Dict:
    """
    Perform comprehensive error analysis with segmentation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features, true labels, and predictions.
    y_true_col : str, default "label"
        Column name for true labels.
    y_pred_col : str, default "predicted"
        Column name for predictions.
    fnr_threshold : float, default 0.3
        FNR threshold to flag as failure segment.
    output_dir : str | Path, optional
        Directory to save outputs.

    Returns
    -------
    dict
        Error analysis results.
    """
    if output_dir is None:
        output_dir = REPORTS_DIR
    else:
        output_dir = Path(output_dir)

    ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("Error Analysis and Segmentation")
    logger.info("=" * 60)

    # Create segments
    logger.info("Creating customer segments...")
    df_segmented = create_segments(df)

    # Analyze errors by segment
    logger.info("Analyzing errors by segment...")
    error_df = analyze_errors_by_segment(
        df_segmented, y_true_col=y_true_col, y_pred_col=y_pred_col
    )

    # Identify failure segments
    logger.info(f"Identifying failure segments (FNR >= {fnr_threshold})...")
    failure_segments = identify_failure_segments(error_df, fnr_threshold=fnr_threshold)

    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_path = plot_error_analysis(error_df, output_dir / "error_analysis.png")

    # Generate report
    logger.info("Generating markdown report...")
    report_path = generate_error_report(
        error_df, failure_segments, output_dir / "error_analysis.md"
    )

    # Compile results
    results = {
        "overall_metrics": error_df[error_df["segment_type"] == "overall"].iloc[0].to_dict(),
        "segment_analysis": error_df[error_df["segment_type"] != "overall"].to_dict("records"),
        "failure_segments": failure_segments.to_dict("records"),
        "outputs": {
            "plot": str(plot_path),
            "report": str(report_path),
        },
    }

    logger.info("=" * 60)
    logger.info("Error analysis completed successfully")
    logger.info(f"  - Failure segments identified: {len(failure_segments)}")
    logger.info(f"  - Report: {report_path}")
    logger.info(f"  - Plot: {plot_path}")
    logger.info("=" * 60)

    return results

