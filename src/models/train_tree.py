"""Train tree-based models (RandomForest, XGBoost, LightGBM) for churn prediction."""

import logging
import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.generate_report import generate_baseline_report
from src.evaluation.metrics import evaluate_model, save_metrics
from src.models.train_baseline import (
    load_features,
    prepare_features,
    temporal_split,
)
from src.utils.config import config
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | dict = "balanced",
    random_state: Optional[int] = None,
) -> Pipeline:
    """
    Create RandomForest model pipeline.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of trees. None means no limit.
    min_samples_split : int, default 2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default 1
        Minimum number of samples required at a leaf node.
    class_weight : str or dict, default "balanced"
        Class weights for handling imbalanced data.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with imputation and RandomForest classifier.
    """
    if random_state is None:
        random_state = config.seed

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline


def create_xgboost_pipeline(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    scale_pos_weight: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Pipeline:
    """
    Create XGBoost model pipeline.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of boosting rounds.
    max_depth : int, default 6
        Maximum tree depth.
    learning_rate : float, default 0.1
        Learning rate (eta).
    scale_pos_weight : float, optional
        Scale for positive class (for imbalanced data).
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with imputation and XGBoost classifier.

    Raises
    ------
    ImportError
        If xgboost is not installed.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        )

    if random_state is None:
        random_state = config.seed

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    return pipeline


def create_lightgbm_pipeline(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    scale_pos_weight: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Pipeline:
    """
    Create LightGBM model pipeline.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of boosting rounds.
    max_depth : int, default 6
        Maximum tree depth.
    learning_rate : float, default 0.1
        Learning rate.
    scale_pos_weight : float, optional
        Scale for positive class (for imbalanced data).
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with imputation and LightGBM classifier.

    Raises
    ------
    ImportError
        If lightgbm is not installed.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "LightGBM is not installed. Install it with: pip install lightgbm"
        )

    if random_state is None:
        random_state = config.seed

    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )

    return pipeline


def train_tree_model(
    model_type: Literal["random_forest", "xgboost", "lightgbm"] = "random_forest",
    features_path: Optional[str | Path] = None,
    model_path: Optional[str | Path] = None,
    train_pct: float = 0.6,
    valid_pct: float = 0.2,
    test_pct: float = 0.2,
    **model_params,
) -> Tuple[Pipeline, dict, dict, dict]:
    """
    Train tree-based model for churn prediction.

    Parameters
    ----------
    model_type : {"random_forest", "xgboost", "lightgbm"}, default "random_forest"
        Type of tree-based model to train.
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
    **model_params
        Additional parameters for the model (passed to pipeline creation).

    Returns
    -------
    Tuple[Pipeline, dict, dict, dict]
        Trained model, train metrics, validation metrics, test metrics.
    """
    logger.info("=" * 60)
    logger.info(f"Training {model_type.replace('_', ' ').title()} Model")
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

    # Calculate scale_pos_weight if needed (for XGBoost/LightGBM)
    if model_type in ["xgboost", "lightgbm"] and "scale_pos_weight" not in model_params:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        if pos_count > 0:
            model_params["scale_pos_weight"] = neg_count / pos_count
            logger.info(f"Calculated scale_pos_weight: {model_params['scale_pos_weight']:.2f}")

    # Create model pipeline
    logger.info(f"Creating {model_type} pipeline...")
    if model_type == "random_forest":
        model = create_random_forest_pipeline(**model_params)
    elif model_type == "xgboost":
        model = create_xgboost_pipeline(**model_params)
    elif model_type == "lightgbm":
        model = create_lightgbm_pipeline(**model_params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train model
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Evaluate on all splits
    train_metrics = evaluate_model(
        y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1], "train"
    )
    valid_metrics = evaluate_model(
        y_valid, model.predict(X_valid), model.predict_proba(X_valid)[:, 1], "validation"
    )
    test_metrics = evaluate_model(
        y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1], "test"
    )

    # Save model
    if model_path is None:
        model_path = MODELS_DIR / f"{model_type}.pkl"
    else:
        model_path = Path(model_path)

    ensure_dir(model_path.parent)
    logger.info(f"Saving model to: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    file_size = model_path.stat().st_size / 1024**2
    logger.info(f"Model saved successfully ({file_size:.2f} MB)")

    logger.info("=" * 60)
    logger.info(f"{model_type.replace('_', ' ').title()} model training completed successfully")
    logger.info("=" * 60)

    return model, train_metrics, valid_metrics, test_metrics


def generate_tree_report(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "random_forest",
    output_path: Optional[str | Path] = None,
) -> dict:
    """
    Generate evaluation report for tree-based model.

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
    model_type : str, default "random_forest"
        Type of model (for report naming).
    output_path : str | Path, optional
        Path to save metrics JSON file.

    Returns
    -------
    dict
        Complete evaluation report.
    """
    if output_path is None:
        output_path = REPORTS_DIR / f"{model_type}_metrics.json"

    logger.info("=" * 60)
    logger.info(f"Generating {model_type.replace('_', ' ').title()} Evaluation Report")
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
        "model": model_type,
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


def compare_models(
    baseline_metrics_path: Optional[str | Path] = None,
    tree_metrics_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Compare baseline and tree-based model metrics.

    Parameters
    ----------
    baseline_metrics_path : str | Path, optional
        Path to baseline metrics JSON file.
    tree_metrics_path : str | Path, optional
        Path to tree model metrics JSON file.
    output_path : str | Path, optional
        Path to save comparison CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with model comparison.
    """
    from src.evaluation.metrics import load_metrics

    if baseline_metrics_path is None:
        baseline_metrics_path = REPORTS_DIR / "baseline_metrics.json"
    if tree_metrics_path is None:
        tree_metrics_path = REPORTS_DIR / "random_forest_metrics.json"
    if output_path is None:
        output_path = REPORTS_DIR / "model_comparison.csv"

    logger.info("=" * 60)
    logger.info("Comparing Models")
    logger.info("=" * 60)

    # Load metrics
    baseline_metrics = load_metrics(baseline_metrics_path)
    tree_metrics = load_metrics(tree_metrics_path)

    # Extract test metrics
    baseline_test = baseline_metrics["splits"]["test"]
    tree_test = tree_metrics["splits"]["test"]

    # Create comparison DataFrame
    comparison_data = [
        {
            "model": "baseline_logistic_regression",
            "split": "test",
            "accuracy": baseline_test["accuracy"],
            "precision": baseline_test["precision"],
            "recall": baseline_test["recall"],
            "f1": baseline_test["f1"],
            "roc_auc": baseline_test["roc_auc"],
            "pr_auc": baseline_test["pr_auc"],
            "false_positive_rate": baseline_test["false_positive_rate"],
        },
        {
            "model": tree_metrics["model"],
            "split": "test",
            "accuracy": tree_test["accuracy"],
            "precision": tree_test["precision"],
            "recall": tree_test["recall"],
            "f1": tree_test["f1"],
            "roc_auc": tree_test["roc_auc"],
            "pr_auc": tree_test["pr_auc"],
            "false_positive_rate": tree_test["false_positive_rate"],
        },
    ]

    df = pd.DataFrame(comparison_data)

    # Save comparison
    ensure_dir(Path(output_path).parent)
    df.to_csv(output_path, index=False)
    logger.info(f"Comparison saved to: {output_path}")

    # Log comparison
    logger.info("\nModel Comparison (Test Set):")
    logger.info(df.to_string(index=False))

    logger.info("=" * 60)

    return df


def main(
    model_type: str = "random_forest",
    features_path: Optional[str] = None,
    model_path: Optional[str] = None,
    report_path: Optional[str] = None,
) -> None:
    """
    Main function to train tree-based model and generate report.

    Parameters
    ----------
    model_type : str, default "random_forest"
        Type of tree-based model to train.
    features_path : str, optional
        Path to features Parquet file.
    model_path : str, optional
        Path to save model.
    report_path : str, optional
        Path to save metrics JSON file.
    """
    # Train model
    model, train_metrics, valid_metrics, test_metrics = train_tree_model(
        model_type=model_type,
        features_path=features_path,
        model_path=model_path,
    )

    # Generate report (reload data for report generation)
    df = load_features(features_path)
    train_df, valid_df, test_df = temporal_split(df)
    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)
    X_test, y_test = prepare_features(test_df)

    generate_tree_report(
        model, X_train, y_train, X_valid, y_valid, X_test, y_test, model_type, report_path
    )

    # Compare with baseline if available
    baseline_path = REPORTS_DIR / "baseline_metrics.json"
    if baseline_path.exists():
        logger.info("\nComparing with baseline model...")
        compare_models(
            baseline_metrics_path=baseline_path,
            tree_metrics_path=report_path or REPORTS_DIR / f"{model_type}_metrics.json",
        )


if __name__ == "__main__":
    import sys

    model_type = sys.argv[1] if len(sys.argv) > 1 else "random_forest"
    features_path = sys.argv[2] if len(sys.argv) > 2 else None
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    report_path = sys.argv[4] if len(sys.argv) > 4 else None

    main(
        model_type=model_type,
        features_path=features_path,
        model_path=model_path,
        report_path=report_path,
    )

