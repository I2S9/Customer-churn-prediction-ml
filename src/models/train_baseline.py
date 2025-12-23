"""Train baseline logistic regression model for churn prediction.

This script implements a simple, interpretable baseline model using:
- Temporal split (train/validation/test)
- Sklearn pipeline: imputation + scaling + logistic regression
- Balanced class weights for imbalanced data
"""

import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"


def temporal_split(df, observation_date_col="observation_date", 
                   train_pct=0.6, valid_pct=0.2, test_pct=0.2):
    """Split dataset temporally based on observation date.
    
    Args:
        df: DataFrame with observation dates
        observation_date_col: Name of observation date column
        train_pct: Percentage for training set
        valid_pct: Percentage for validation set
        test_pct: Percentage for test set
    
    Returns:
        train_df, valid_df, test_df
    """
    if observation_date_col not in df.columns:
        raise ValueError(f"Column '{observation_date_col}' not found in dataset")
    
    # Ensure observation_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[observation_date_col]):
        df[observation_date_col] = pd.to_datetime(df[observation_date_col])
    
    # Sort by observation date
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
    """Prepare feature matrix X and target vector y.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
    
    Returns:
        X, y, feature_names
    """
    if exclude_cols is None:
        exclude_cols = ["customer_id", "observation_date", target_col]
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError("No feature columns found")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y, feature_cols


def create_baseline_pipeline():
    """Create sklearn pipeline for baseline model."""
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)),
    ])
    return pipeline


def evaluate_model(y_true, y_pred, y_proba=None, set_name=""):
    """Calculate and return evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    
    # Class distribution
    metrics["class_distribution"] = {
        "negative": int((y_true == 0).sum()),
        "positive": int((y_true == 1).sum()),
    }
    
    return metrics


def save_model(model, filepath):
    """Save trained model to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def save_metrics(metrics, filepath):
    """Save metrics to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {filepath}")


def main():
    """Main function to train baseline model."""
    parser = argparse.ArgumentParser(
        description="Train baseline logistic regression model for churn prediction"
    )
    parser.add_argument(
        "features_file",
        type=Path,
        help="Path to feature CSV file",
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
        "--model-output",
        type=Path,
        default=MODELS_DIR / "baseline.pkl",
        help="Output path for trained model (default: models/baseline.pkl)",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=RESULTS_DIR / "baseline_metrics.json",
        help="Output path for metrics (default: results/metrics/baseline_metrics.json)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Baseline Model Training")
    print("=" * 60)
    print(f"Features file: {args.features_file}")
    print(f"Target column: {args.target_col}")
    print(f"Observation date column: {args.observation_date_col}")
    print(f"Train/Valid/Test split: {args.train_pct:.1%}/{args.valid_pct:.1%}/{args.test_pct:.1%}")
    print("-" * 60)
    
    # Load data
    print("\nLoading features...")
    if not args.features_file.exists():
        print(f"Error: Features file not found: {args.features_file}")
        sys.exit(1)
    
    df = pd.read_csv(args.features_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check target column
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
    print(f"Feature names: {', '.join(feature_names[:5])}...")
    
    # Check class distribution
    print(f"\nClass distribution (train):")
    print(f"  Negative (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"  Positive (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    
    # Create and train model
    print("\nCreating baseline pipeline...")
    model = create_baseline_pipeline()
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_valid_pred = model.predict(X_valid)
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    valid_metrics = evaluate_model(y_valid, y_valid_pred, y_valid_proba, "validation")
    
    print(f"  Accuracy: {valid_metrics['accuracy']:.4f}")
    print(f"  Precision: {valid_metrics['precision']:.4f}")
    print(f"  Recall: {valid_metrics['recall']:.4f}")
    print(f"  F1: {valid_metrics['f1']:.4f}")
    if valid_metrics['roc_auc']:
        print(f"  ROC-AUC: {valid_metrics['roc_auc']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, "test")
    
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    if test_metrics['roc_auc']:
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    # Save model and metrics
    print("\nSaving model and metrics...")
    save_model(model, args.model_output)
    
    all_metrics = {
        "model": "baseline_logistic_regression",
        "timestamp": datetime.now().isoformat(),
        "features": feature_names,
        "n_features": len(feature_names),
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "validation": valid_metrics,
        "test": test_metrics,
    }
    
    save_metrics(all_metrics, args.metrics_output)
    
    # Print classification report
    print("\n" + "=" * 60)
    print("Test Set Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_test_pred, target_names=["No Churn", "Churn"]))
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

