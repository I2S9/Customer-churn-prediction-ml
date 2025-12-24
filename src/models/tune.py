"""Controlled hyperparameter tuning with RandomizedSearchCV."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def get_param_grid(model_type: str) -> Dict[str, Any]:
    """
    Get parameter grid for hyperparameter tuning.

    Parameters
    ----------
    model_type : str
        Type of model ("logistic_regression" or "random_forest").

    Returns
    -------
    dict
        Parameter grid for RandomizedSearchCV.
    """
    if model_type == "logistic_regression":
        return {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
            "clf__max_iter": [100, 200, 300],
        }
    elif model_type == "random_forest":
        return {
            "clf__n_estimators": [50, 100, 200, 300],
            "clf__max_depth": [3, 5, 7, 10, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_pipeline(model_type: str) -> Pipeline:
    """
    Create model pipeline for tuning.

    Parameters
    ----------
    model_type : str
        Type of model ("logistic_regression" or "random_forest").

    Returns
    -------
    sklearn.pipeline.Pipeline
        Model pipeline.
    """
    if model_type == "logistic_regression":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=config.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    elif model_type == "random_forest":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=config.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def tune_model(
    model_type: str = "random_forest",
    features_path: Optional[str | Path] = None,
    n_iter: int = 30,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning with RandomizedSearchCV.

    Parameters
    ----------
    model_type : str, default "random_forest"
        Type of model to tune.
    features_path : str | Path, optional
        Path to features Parquet file.
    n_iter : int, default 30
        Number of parameter settings sampled (budget).
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, default "roc_auc"
        Scoring metric for model selection.
    random_state : int, optional
        Random state for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs.

    Returns
    -------
    dict
        Dictionary containing tuning results and best model.
    """
    if random_state is None:
        random_state = config.seed

    logger.info("=" * 60)
    logger.info(f"Hyperparameter Tuning: {model_type.replace('_', ' ').title()}")
    logger.info("=" * 60)
    logger.info(f"Budget: {n_iter} configurations")
    logger.info(f"CV folds: {cv}")
    logger.info(f"Scoring: {scoring}")
    logger.info(f"Random state: {random_state}")

    # Load and prepare data
    df = load_features(features_path)
    train_df, valid_df, test_df = temporal_split(df)

    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)
    X_test, y_test = prepare_features(test_df)

    # Combine train and validation for CV
    X_cv = pd.concat([X_train, X_valid], axis=0)
    y_cv = pd.concat([y_train, y_valid], axis=0)

    logger.info(f"CV data: {len(X_cv):,} samples")
    logger.info(f"Test data: {len(X_test):,} samples")

    # Create pipeline and parameter grid
    pipeline = create_pipeline(model_type)
    param_grid = get_param_grid(model_type)

    logger.info(f"Parameter grid size: {len(param_grid)} parameters")
    for param, values in param_grid.items():
        logger.info(f"  {param}: {values}")

    # Perform randomized search
    logger.info("\nStarting RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    random_search.fit(X_cv, y_cv)

    logger.info("\nTuning completed!")
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best CV score ({scoring}): {random_search.best_score_:.4f}")

    # Evaluate best model on test set
    best_model = random_search.best_estimator_
    test_metrics = evaluate_model(
        y_test,
        best_model.predict(X_test),
        best_model.predict_proba(X_test)[:, 1],
        split_name="test",
    )

    # Collect results
    results = {
        "model_type": model_type,
        "tuning_config": {
            "n_iter": n_iter,
            "cv": cv,
            "scoring": scoring,
            "random_state": random_state,
        },
        "best_params": {k: str(v) for k, v in random_search.best_params_.items()},
        "best_cv_score": float(random_search.best_score_),
        "test_metrics": test_metrics["metrics"],
        "cv_results": {
            "mean_test_score": random_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": random_search.cv_results_["std_test_score"].tolist(),
            "mean_train_score": random_search.cv_results_["mean_train_score"].tolist(),
            "std_train_score": random_search.cv_results_["std_train_score"].tolist(),
            "params": [
                {k: str(v) for k, v in params.items()}
                for params in random_search.cv_results_["params"]
            ],
        },
    }

    return {
        "results": results,
        "best_model": best_model,
        "random_search": random_search,
    }


def save_tuned_model(
    model: Pipeline,
    model_path: Optional[str | Path] = None,
    model_type: str = "random_forest",
) -> Path:
    """
    Save tuned model to disk.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Best tuned model.
    model_path : str | Path, optional
        Path to save model. Defaults to models/{model_type}_tuned.pkl.
    model_type : str, default "random_forest"
        Type of model (for default path).

    Returns
    -------
    Path
        Path where model was saved.
    """
    if model_path is None:
        model_path = MODELS_DIR / f"{model_type}_tuned.pkl"
    else:
        model_path = Path(model_path)

    ensure_dir(model_path.parent)

    logger.info(f"Saving tuned model to: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    file_size = model_path.stat().st_size / 1024**2
    logger.info(f"Tuned model saved successfully ({file_size:.2f} MB)")

    return model_path


def save_tuning_summary(
    results: Dict[str, Any],
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Save tuning summary to JSON file.

    Parameters
    ----------
    results : dict
        Tuning results dictionary.
    output_path : str | Path, optional
        Path to save summary JSON file.

    Returns
    -------
    Path
        Path where summary was saved.
    """
    if output_path is None:
        model_type = results["model_type"]
        output_path = REPORTS_DIR / f"tuning_summary_{model_type}.json"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    logger.info(f"Saving tuning summary to: {output_path}")

    # Convert numpy types to native Python types
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

    serializable_results = convert_to_json_serializable(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / 1024**2
    logger.info(f"Tuning summary saved successfully ({file_size:.4f} MB)")

    return output_path


def main(
    model_type: str = "random_forest",
    features_path: Optional[str] = None,
    n_iter: int = 30,
    cv: int = 5,
    scoring: str = "roc_auc",
    model_path: Optional[str] = None,
    summary_path: Optional[str] = None,
) -> None:
    """
    Main function to perform hyperparameter tuning.

    Parameters
    ----------
    model_type : str, default "random_forest"
        Type of model to tune.
    features_path : str, optional
        Path to features Parquet file.
    n_iter : int, default 30
        Number of parameter configurations to try.
    cv : int, default 5
        Number of cross-validation folds.
    scoring : str, default "roc_auc"
        Scoring metric for model selection.
    model_path : str, optional
        Path to save tuned model.
    summary_path : str, optional
        Path to save tuning summary JSON.
    """
    # Perform tuning
    tuning_output = tune_model(
        model_type=model_type,
        features_path=features_path,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
    )

    results = tuning_output["results"]
    best_model = tuning_output["best_model"]

    # Save model
    save_tuned_model(best_model, model_path=model_path, model_type=model_type)

    # Save summary
    save_tuning_summary(results, output_path=summary_path)

    logger.info("=" * 60)
    logger.info("Hyperparameter tuning completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys

    model_type = sys.argv[1] if len(sys.argv) > 1 else "random_forest"
    features_path = sys.argv[2] if len(sys.argv) > 2 else None
    n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    cv = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    scoring = sys.argv[5] if len(sys.argv) > 5 else "roc_auc"
    model_path = sys.argv[6] if len(sys.argv) > 6 else None
    summary_path = sys.argv[7] if len(sys.argv) > 7 else None

    main(
        model_type=model_type,
        features_path=features_path,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        model_path=model_path,
        summary_path=summary_path,
    )

