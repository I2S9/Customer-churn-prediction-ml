"""Train tree-based model for churn prediction.

This script implements a tree-based model (RandomForest or XGBoost) for comparison
with the baseline logistic regression model.
"""

import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM (optional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

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
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
REPORTS_DIR = PROJECT_ROOT / "reports"


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


def create_random_forest_model(n_estimators=100, max_depth=10, 
                               class_weight="balanced", random_state=42):
    """Create RandomForest model with pipeline."""
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    return pipeline


def create_xgboost_model(n_estimators=100, max_depth=6, 
                        scale_pos_weight=None, random_state=42):
    """Create XGBoost model with pipeline."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        # Will be calculated from training data
        scale_pos_weight = 1.0
    
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
        )),
    ])
    return pipeline


def create_lightgbm_model(n_estimators=100, max_depth=6,
                          scale_pos_weight=None, random_state=42):
    """Create LightGBM model with pipeline."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
    
    if scale_pos_weight is None:
        scale_pos_weight = 1.0
    
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            verbose=-1,
        )),
    ])
    return pipeline


def calculate_scale_pos_weight(y_train):
    """Calculate scale_pos_weight for imbalanced data."""
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    
    if positive_count == 0:
        return 1.0
    
    return negative_count / positive_count


def main():
    """Main function to train tree-based model."""
    parser = argparse.ArgumentParser(
        description="Train tree-based model for churn prediction"
    )
    parser.add_argument(
        "features_file",
        type=Path,
        help="Path to feature CSV file",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost", "lightgbm"],
        default="random_forest",
        help="Type of tree model (default: random_forest)",
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
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum tree depth (default: 10)",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Output path for trained model (default: models/{model_type}.pkl)",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Output path for metrics (default: reports/{model_type}_metrics.json)",
    )
    args = parser.parse_args()
    
    # Set default output paths
    if args.model_output is None:
        args.model_output = MODELS_DIR / f"{args.model_type}.pkl"
    if args.metrics_output is None:
        args.metrics_output = REPORTS_DIR / f"{args.model_type}_metrics.json"
    
    print("=" * 60)
    print("Tree-Based Model Training")
    print("=" * 60)
    print(f"Features file: {args.features_file}")
    print(f"Model type: {args.model_type}")
    print(f"Target column: {args.target_col}")
    print(f"Train/Valid/Test split: {args.train_pct:.1%}/{args.valid_pct:.1%}/{args.test_pct:.1%}")
    print(f"n_estimators: {args.n_estimators}, max_depth: {args.max_depth}")
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
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    try:
        if args.model_type == "random_forest":
            model = create_random_forest_model(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=42,
            )
        elif args.model_type == "xgboost":
            scale_pos_weight = calculate_scale_pos_weight(y_train)
            model = create_xgboost_model(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
            )
        elif args.model_type == "lightgbm":
            scale_pos_weight = calculate_scale_pos_weight(y_train)
            model = create_lightgbm_model(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
            )
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_valid_pred = model.predict(X_valid)
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    valid_report = evaluate_model(y_valid, y_valid_pred, y_valid_proba)
    
    print_metrics_summary(valid_report["metrics"], set_name="Validation")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_report = evaluate_model(y_test, y_test_pred, y_test_proba)
    
    print_metrics_summary(test_report["metrics"], set_name="Test")
    
    # Save model
    print("\nSaving model...")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_output, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {args.model_output}")
    
    # Save metrics - save test set evaluation as main report
    print("\nSaving evaluation report...")
    
    # Add validation metrics to test report
    test_report["validation_metrics"] = valid_report["metrics"]
    
    save_evaluation_report(
        test_report,
        args.metrics_output,
        model_name=args.model_type,
        dataset_name="test",
        additional_metadata={
            "n_features": len(feature_names),
            "train_size": len(train_df),
            "valid_size": len(valid_df),
            "test_size": len(test_df),
            "features": feature_names,
            "model_params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "model_type": args.model_type,
            },
        },
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel: {args.model_output}")
    print(f"Metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()

