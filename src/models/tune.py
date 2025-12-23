"""Hyperparameter tuning with RandomizedSearchCV.

This script performs controlled hyperparameter tuning with:
- Fixed seed for reproducibility
- Limited budget (default: 30 configurations)
- RandomizedSearchCV for efficient search
"""

import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, randint, loguniform

from src.evaluation.metrics import (
    evaluate_model,
    save_evaluation_report,
    print_metrics_summary,
)

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Fixed seed for reproducibility
RANDOM_STATE = 42


def temporal_split(df, observation_date_col="observation_date", 
                   train_pct=0.6, valid_pct=0.2, test_pct=0.2):
    """Split dataset temporally based on observation date."""
    if observation_date_col not in df.columns:
        raise ValueError(f"Column '{observation_date_col}' not found in dataset")
    
    if not pd.api.types.is_datetime64_any_dtype(df[observation_date_col]):
        df[observation_date_col] = pd.to_datetime(df[observation_date_col])
    
    df_sorted = df.sort_values(observation_date_col).reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * train_pct)
    valid_end = int(n * (train_pct + valid_pct))
    
    train_df = df_sorted.iloc[:train_end].copy()
    valid_df = df_sorted.iloc[train_end:valid_end].copy()
    test_df = df_sorted.iloc[valid_end:].copy()
    
    return train_df, valid_df, test_df


def prepare_features_and_target(df, target_col="churn", 
                                exclude_cols=None):
    """Prepare feature matrix X and target vector y."""
    if exclude_cols is None:
        exclude_cols = ["customer_id", "observation_date", target_col]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError("No feature columns found")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y, feature_cols


def get_param_distributions(model_type):
    """Get parameter distributions for RandomizedSearchCV.
    
    Args:
        model_type: Type of model ('logistic' or 'random_forest')
    
    Returns:
        Dictionary of parameter distributions
    """
    if model_type == "logistic":
        return {
            "impute__strategy": ["mean", "median"],
            "clf__C": loguniform(1e-3, 1e2),
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
            "clf__max_iter": [200, 500, 1000],
        }
    elif model_type == "random_forest":
        return {
            "impute__strategy": ["mean", "median"],
            "clf__n_estimators": randint(50, 300),
            "clf__max_depth": randint(5, 30),
            "clf__min_samples_split": randint(2, 20),
            "clf__min_samples_leaf": randint(1, 10),
            "clf__max_features": ["sqrt", "log2", None],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_base_pipeline(model_type):
    """Create base pipeline for the model type."""
    if model_type == "logistic":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)),
        ])
    elif model_type == "random_forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_tuning_summary(search_results, filepath, model_type, n_iter, cv):
    """Save tuning summary to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract top results
    results_df = pd.DataFrame(search_results.cv_results_)
    top_results = results_df.nlargest(5, "mean_test_score")[
        ["mean_test_score", "std_test_score", "params"]
    ].to_dict("records")
    
    summary = {
        "metadata": {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "n_iter": n_iter,
            "cv": cv,
            "random_state": RANDOM_STATE,
        },
        "best_params": search_results.best_params_,
        "best_score": float(search_results.best_score_),
        "best_std": float(search_results.cv_results_["std_test_score"][search_results.best_index_]),
        "top_5_configs": top_results,
        "all_results": {
            "mean_scores": search_results.cv_results_["mean_test_score"].tolist(),
            "std_scores": search_results.cv_results_["std_test_score"].tolist(),
        },
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Tuning summary saved to: {filepath}")


def main():
    """Main function for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with RandomizedSearchCV"
    )
    parser.add_argument(
        "features_file",
        type=Path,
        help="Path to feature CSV file",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest"],
        default="random_forest",
        help="Type of model to tune (default: random_forest)",
    )
    parser.add_argument(
        "--target-col",
        default="churn",
        help="Name of target column (default: churn)",
    )
    parser.add_argument(
        "--observation-date-col",
        default="observation_date",
        help="Name of observation date column (default: observation_date)",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.6,
        help="Training set percentage (default: 0.6)",
    )
    parser.add_argument(
        "--valid-pct",
        type=float,
        default=0.2,
        help="Validation set percentage (default: 0.2)",
    )
    parser.add_argument(
        "--test-pct",
        type=float,
        default=0.2,
        help="Test set percentage (default: 0.2)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of parameter configurations to try (default: 30)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--scoring",
        default="roc_auc",
        help="Scoring metric for optimization (default: roc_auc)",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Output path for best model (default: models/{model_type}_tuned.pkl)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Output path for tuning summary (default: reports/tuning_summary.json)",
    )
    args = parser.parse_args()
    
    # Set default output paths
    if args.model_output is None:
        args.model_output = MODELS_DIR / f"{args.model_type}_tuned.pkl"
    if args.summary_output is None:
        args.summary_output = REPORTS_DIR / "tuning_summary.json"
    
    print("=" * 60)
    print("Hyperparameter Tuning")
    print("=" * 60)
    print(f"Features file: {args.features_file}")
    print(f"Model type: {args.model_type}")
    print(f"Budget: {args.n_iter} configurations")
    print(f"CV folds: {args.cv}")
    print(f"Scoring: {args.scoring}")
    print(f"Random state: {RANDOM_STATE}")
    print("-" * 60)
    
    # Load data
    print("\nLoading features...")
    if not args.features_file.exists():
        print(f"Error: Features file not found: {args.features_file}")
        sys.exit(1)
    
    df = pd.read_csv(args.features_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if args.target_col not in df.columns:
        print(f"Error: Target column '{args.target_col}' not found")
        sys.exit(1)
    
    # Temporal split
    print("\nPerforming temporal split...")
    train_df, valid_df, test_df = temporal_split(
        df,
        observation_date_col=args.observation_date_col,
        train_pct=args.train_pct,
        valid_pct=args.valid_pct,
        test_pct=args.test_pct,
    )
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(valid_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    # Prepare features and targets
    print("\nPreparing features and targets...")
    X_train, y_train, feature_names = prepare_features_and_target(
        train_df, target_col=args.target_col
    )
    X_valid, y_valid, _ = prepare_features_and_target(
        valid_df, target_col=args.target_col
    )
    X_test, y_test, _ = prepare_features_and_target(
        test_df, target_col=args.target_col
    )
    
    print(f"Features: {len(feature_names)}")
    
    # Check class distribution
    print(f"\nClass distribution (train):")
    print(f"  Negative (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  Positive (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    
    # Create base pipeline
    print(f"\nCreating base {args.model_type} pipeline...")
    base_pipeline = create_base_pipeline(args.model_type)
    
    # Get parameter distributions
    param_dist = get_param_distributions(args.model_type)
    
    print(f"Parameter space: {len(param_dist)} hyperparameters")
    print(f"Search budget: {args.n_iter} configurations")
    
    # Perform randomized search
    print("\nStarting RandomizedSearchCV...")
    print("-" * 60)
    
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    
    print("\n" + "=" * 60)
    print("Tuning Complete")
    print("=" * 60)
    print(f"Best {args.scoring}: {search.best_score_:.4f} (+/- {search.cv_results_['std_test_score'][search.best_index_]:.4f})")
    print(f"\nBest parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Evaluate best model on validation set
    print("\nEvaluating best model on validation set...")
    best_model = search.best_estimator_
    y_valid_pred = best_model.predict(X_valid)
    y_valid_proba = best_model.predict_proba(X_valid)[:, 1]
    valid_report = evaluate_model(y_valid, y_valid_pred, y_valid_proba)
    
    print_metrics_summary(valid_report["metrics"], set_name="Validation")
    
    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    test_report = evaluate_model(y_test, y_test_pred, y_test_proba)
    
    print_metrics_summary(test_report["metrics"], set_name="Test")
    
    # Save best model
    print("\nSaving best model...")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_output, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to: {args.model_output}")
    
    # Save tuning summary
    print("\nSaving tuning summary...")
    save_tuning_summary(
        search,
        args.summary_output,
        args.model_type,
        args.n_iter,
        args.cv,
    )
    
    # Save evaluation report
    eval_output = REPORTS_DIR / f"{args.model_type}_tuned_metrics.json"
    test_report["validation_metrics"] = valid_report["metrics"]
    test_report["tuning_info"] = {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "n_iter": args.n_iter,
        "cv": args.cv,
        "scoring": args.scoring,
    }
    
    from src.evaluation.metrics import save_evaluation_report
    save_evaluation_report(
        test_report,
        eval_output,
        model_name=f"{args.model_type}_tuned",
        dataset_name="test",
        additional_metadata={
            "n_features": len(feature_names),
            "train_size": len(train_df),
            "valid_size": len(valid_df),
            "test_size": len(test_df),
            "features": feature_names,
        },
    )
    
    print("\n" + "=" * 60)
    print("Tuning complete!")
    print("=" * 60)
    print(f"\nBest model: {args.model_output}")
    print(f"Tuning summary: {args.summary_output}")
    print(f"Evaluation report: {eval_output}")


if __name__ == "__main__":
    main()

