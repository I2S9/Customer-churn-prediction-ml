"""Generate evaluation reports for models."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import evaluate_model, save_metrics
from src.utils.paths import REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_baseline_report(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Optional[str | Path] = None,
) -> Dict:
    """
    Generate comprehensive evaluation report for baseline model.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline.
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training labels.
    X_valid : pandas.DataFrame
        Validation features.
    y_valid : pandas.Series
        Validation labels.
    X_test : pandas.DataFrame
        Test features.
    y_test : pandas.Series
        Test labels.
    output_path : str | Path, optional
        Path to save metrics JSON file. Defaults to reports/baseline_metrics.json.

    Returns
    -------
    dict
        Complete evaluation report with all splits.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "baseline_metrics.json"

    logger.info("=" * 60)
    logger.info("Generating Baseline Model Evaluation Report")
    logger.info("=" * 60)

    # Evaluate on all splits
    train_results = evaluate_model(
        y_train,
        model.predict(X_train),
        model.predict_proba(X_train)[:, 1],
        split_name="train",
    )

    valid_results = evaluate_model(
        y_valid,
        model.predict(X_valid),
        model.predict_proba(X_valid)[:, 1],
        split_name="validation",
    )

    test_results = evaluate_model(
        y_test,
        model.predict(X_test),
        model.predict_proba(X_test)[:, 1],
        split_name="test",
    )

    # Combine all results
    report = {
        "model": "baseline_logistic_regression",
        "splits": {
            "train": train_results["metrics"],
            "validation": valid_results["metrics"],
            "test": test_results["metrics"],
        },
    }

    # Save report
    save_metrics(report, output_path)

    logger.info("=" * 60)
    logger.info("Evaluation report generated successfully")
    logger.info("=" * 60)

    return report


def main(
    model_path: Optional[str] = None,
    features_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Main function to generate baseline evaluation report.

    Parameters
    ----------
    model_path : str, optional
        Path to saved model pickle file.
    features_path : str, optional
        Path to features Parquet file.
    output_path : str, optional
        Path to save metrics JSON file.
    """
    import pickle

    from src.models import load_features, prepare_features, temporal_split

    # Load model
    if model_path is None:
        from src.utils.paths import MODELS_DIR

        model_path = MODELS_DIR / "baseline.pkl"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load features and split
    df = load_features(features_path)
    train_df, valid_df, test_df = temporal_split(df)

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)
    X_test, y_test = prepare_features(test_df)

    # Generate report
    generate_baseline_report(
        model, X_train, y_train, X_valid, y_valid, X_test, y_test, output_path=output_path
    )


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    features_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    main(model_path=model_path, features_path=features_path, output_path=output_path)

