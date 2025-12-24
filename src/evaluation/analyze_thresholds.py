"""Utility to perform threshold analysis on trained models."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.thresholds import perform_threshold_analysis
from src.models.train_baseline import load_features, prepare_features, temporal_split
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_model_thresholds(
    model_path: str | Path,
    features_path: Optional[str | Path] = None,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    split: str = "test",
    output_dir: Optional[str | Path] = None,
) -> dict:
    """
    Perform threshold analysis on a trained model.

    Parameters
    ----------
    model_path : str | Path
        Path to saved model pickle file.
    features_path : str | Path, optional
        Path to features Parquet file.
    cost_fp : float, default 1.0
        Cost per false positive.
    cost_fn : float, default 5.0
        Cost per false negative.
    split : str, default "test"
        Data split to analyze ("train", "validation", or "test").
    output_dir : str | Path, optional
        Directory to save outputs.

    Returns
    -------
    dict
        Threshold analysis results.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if features_path is None:
        features_path = PROCESSED_DATA_DIR / "features.parquet"
    else:
        features_path = Path(features_path)

    if output_dir is None:
        output_dir = REPORTS_DIR
    else:
        output_dir = Path(output_dir)

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

    # Perform threshold analysis
    results = perform_threshold_analysis(
        y_split, y_pred_proba, cost_fp=cost_fp, cost_fn=cost_fn, output_dir=output_dir
    )

    return results


def main(
    model_path: Optional[str] = None,
    features_path: Optional[str] = None,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    split: str = "test",
) -> None:
    """
    Main function to perform threshold analysis.

    Parameters
    ----------
    model_path : str, optional
        Path to saved model. Defaults to models/baseline.pkl.
    features_path : str, optional
        Path to features Parquet file.
    cost_fp : float, default 1.0
        Cost per false positive.
    cost_fn : float, default 5.0
        Cost per false negative.
    split : str, default "test"
        Data split to analyze.
    """
    if model_path is None:
        model_path = MODELS_DIR / "baseline.pkl"

    analyze_model_thresholds(
        model_path=model_path,
        features_path=features_path,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        split=split,
    )


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    features_path = sys.argv[2] if len(sys.argv) > 2 else None
    cost_fp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    cost_fn = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    split = sys.argv[5] if len(sys.argv) > 5 else "test"

    main(
        model_path=model_path,
        features_path=features_path,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        split=split,
    )

