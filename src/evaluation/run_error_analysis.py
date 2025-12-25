"""Run error analysis on trained models."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from src.evaluation.error_analysis import perform_error_analysis
from src.models.train_baseline import load_features, prepare_features, temporal_split
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_error_analysis(
    model_path: str | Path,
    features_path: Optional[str | Path] = None,
    split: str = "test",
    fnr_threshold: float = 0.3,
    threshold: float = 0.5,
) -> dict:
    """
    Run error analysis on a trained model.

    Parameters
    ----------
    model_path : str | Path
        Path to saved model pickle file.
    features_path : str | Path, optional
        Path to features Parquet file.
    split : str, default "test"
        Data split to analyze ("train", "validation", or "test").
    fnr_threshold : float, default 0.3
        FNR threshold to flag as failure segment.
    threshold : float, default 0.5
        Classification threshold.

    Returns
    -------
    dict
        Error analysis results.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if features_path is None:
        features_path = PROCESSED_DATA_DIR / "features.parquet"
    else:
        features_path = Path(features_path)

    logger.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loading features from: {features_path}")
    df = load_features(features_path)

    # Split data
    train_df, valid_df, test_df = temporal_split(df)

    # Select split
    if split == "train":
        split_df = train_df
    elif split == "validation":
        split_df = valid_df
    elif split == "test":
        split_df = test_df
    else:
        raise ValueError(f"Unknown split: {split}")

    # Prepare features
    X_split, y_split = prepare_features(split_df)

    # Get predictions
    logger.info(f"Generating predictions for {split} set...")
    y_pred_proba = model.predict_proba(X_split)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Combine with original features
    df_analysis = split_df.copy()
    df_analysis["predicted"] = y_pred
    df_analysis["predicted_proba"] = y_pred_proba

    # Perform error analysis
    results = perform_error_analysis(
        df_analysis,
        y_true_col="label",
        y_pred_col="predicted",
        fnr_threshold=fnr_threshold,
    )

    return results


def main(
    model_path: Optional[str] = None,
    features_path: Optional[str] = None,
    split: str = "test",
    fnr_threshold: float = 0.3,
    threshold: float = 0.5,
) -> None:
    """
    Main function to run error analysis.

    Parameters
    ----------
    model_path : str, optional
        Path to saved model. Defaults to models/baseline.pkl.
    features_path : str, optional
        Path to features Parquet file.
    split : str, default "test"
        Data split to analyze.
    fnr_threshold : float, default 0.3
        FNR threshold to flag as failure segment.
    threshold : float, default 0.5
        Classification threshold.
    """
    if model_path is None:
        model_path = MODELS_DIR / "baseline.pkl"

    run_error_analysis(
        model_path=model_path,
        features_path=features_path,
        split=split,
        fnr_threshold=fnr_threshold,
        threshold=threshold,
    )


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    features_path = sys.argv[2] if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    fnr_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

    main(
        model_path=model_path,
        features_path=features_path,
        split=split,
        fnr_threshold=fnr_threshold,
        threshold=threshold,
    )

