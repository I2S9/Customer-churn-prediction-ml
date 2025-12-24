"""Train baseline logistic regression model for churn prediction."""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import config
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_features(features_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load features from Parquet file.

    Parameters
    ----------
    features_path : str | Path, optional
        Path to features Parquet file. Defaults to data/processed/features.parquet.

    Returns
    -------
    pandas.DataFrame
        Features DataFrame.
    """
    if features_path is None:
        features_path = PROCESSED_DATA_DIR / "features.parquet"
    else:
        features_path = Path(features_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    logger.info(f"Loading features from: {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df):,} samples with {len(df.columns)} columns")

    return df


def temporal_split(
    df: pd.DataFrame,
    train_pct: float = 0.6,
    valid_pct: float = 0.2,
    test_pct: float = 0.2,
    date_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (respecting time order).

    Parameters
    ----------
    df : pandas.DataFrame
        Features DataFrame.
    train_pct : float, default 0.6
        Percentage of data for training.
    valid_pct : float, default 0.2
        Percentage of data for validation.
    test_pct : float, default 0.2
        Percentage of data for testing.
    date_col : str, optional
        Column name to use for temporal ordering. If None, uses index order.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames.
    """
    if abs(train_pct + valid_pct + test_pct - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test percentages must sum to 1.0")

    logger.info("Performing temporal split...")

    # Sort by date if date column provided, otherwise by index
    if date_col and date_col in df.columns:
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        logger.info(f"Sorted by {date_col}")
    else:
        df_sorted = df.reset_index(drop=True)
        logger.info("Sorted by index order")

    n_total = len(df_sorted)
    n_train = int(n_total * train_pct)
    n_valid = int(n_total * valid_pct)

    train_df = df_sorted.iloc[:n_train].copy()
    valid_df = df_sorted.iloc[n_train : n_train + n_valid].copy()
    test_df = df_sorted.iloc[n_train + n_valid :].copy()

    logger.info(
        f"Split: train={len(train_df):,} ({len(train_df)/n_total*100:.1f}%), "
        f"valid={len(valid_df):,} ({len(valid_df)/n_total*100:.1f}%), "
        f"test={len(test_df):,} ({len(test_df)/n_total*100:.1f}%)"
    )

    return train_df, valid_df, test_df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for modeling.

    Parameters
    ----------
    df : pandas.DataFrame
        Features DataFrame.
    target_col : str, default "label"
        Name of target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (X) and target (y).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # Exclude non-feature columns
    exclude_cols = ["customer_id", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info(f"Prepared {len(feature_cols)} features for {len(X):,} samples")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


def create_baseline_pipeline() -> Pipeline:
    """
    Create baseline model pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with imputation, scaling, and logistic regression.
    """
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    random_state=config.seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "test",
) -> dict:
    """
    Evaluate model and return metrics.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline.
    X : pandas.DataFrame
        Features.
    y : pandas.Series
        True labels.
    split_name : str, default "test"
        Name of the split being evaluated.

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    logger.info(f"Evaluating on {split_name} set...")

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Additional metrics
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * (precision_score * recall_score) / (precision_score + recall_score)
        if (precision_score + recall_score) > 0
        else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    logger.info(f"{split_name.upper()} Metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC: {pr_auc:.4f}")
    logger.info(f"  Precision: {precision_score:.4f}")
    logger.info(f"  Recall: {recall_score:.4f}")
    logger.info(f"  F1-Score: {f1_score:.4f}")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y, y_pred))

    return metrics


def save_model(model: Pipeline, model_path: Optional[str | Path] = None) -> Path:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model.
    model_path : str | Path, optional
        Path to save model. Defaults to models/baseline.pkl.

    Returns
    -------
    Path
        Path where model was saved.
    """
    if model_path is None:
        model_path = MODELS_DIR / "baseline.pkl"
    else:
        model_path = Path(model_path)

    ensure_dir(model_path.parent)

    logger.info(f"Saving model to: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    file_size = model_path.stat().st_size / 1024**2
    logger.info(f"Model saved successfully ({file_size:.2f} MB")

    return model_path


def train_baseline(
    features_path: Optional[str | Path] = None,
    model_path: Optional[str | Path] = None,
    train_pct: float = 0.6,
    valid_pct: float = 0.2,
    test_pct: float = 0.2,
) -> Tuple[Pipeline, dict, dict, dict]:
    """
    Train baseline logistic regression model.

    Parameters
    ----------
    features_path : str | Path, optional
        Path to features Parquet file.
    model_path : str | Path, optional
        Path to save model.
    train_pct : float, default 0.6
        Percentage of data for training.
    valid_pct : float, default 0.2
        Percentage of data for validation.
    test_pct : float, default 0.2
        Percentage of data for testing.

    Returns
    -------
    Tuple[Pipeline, dict, dict, dict]
        Trained model, train metrics, validation metrics, test metrics.
    """
    logger.info("=" * 60)
    logger.info("Training Baseline Model (Logistic Regression)")
    logger.info("=" * 60)

    # Load features
    df = load_features(features_path)

    # Temporal split
    train_df, valid_df, test_df = temporal_split(
        df, train_pct=train_pct, valid_pct=valid_pct, test_pct=test_pct
    )

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)
    X_test, y_test = prepare_features(test_df)

    # Create and train model
    logger.info("Creating baseline pipeline...")
    model = create_baseline_pipeline()

    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Evaluate on all splits
    train_metrics = evaluate_model(model, X_train, y_train, "train")
    valid_metrics = evaluate_model(model, X_valid, y_valid, "validation")
    test_metrics = evaluate_model(model, X_test, y_test, "test")

    # Save model
    save_model(model, model_path)

    logger.info("=" * 60)
    logger.info("Baseline model training completed successfully")
    logger.info("=" * 60)

    return model, train_metrics, valid_metrics, test_metrics


def main(
    features_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> None:
    """
    Main function to train baseline model.

    Parameters
    ----------
    features_path : str, optional
        Path to features Parquet file.
    model_path : str, optional
        Path to save model.
    """
    train_baseline(features_path=features_path, model_path=model_path)


if __name__ == "__main__":
    import sys

    features_path = sys.argv[1] if len(sys.argv) > 1 else None
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    main(features_path=features_path, model_path=model_path)

