"""Evaluate a trained model using the standardized evaluation framework.

This script loads a trained model and evaluates it on test data,
generating a comprehensive JSON report.
"""

import sys
import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

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


def load_model(model_path: Path):
    """Load trained model from pickle file."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def prepare_test_data(features_file: Path, target_col: str = "churn",
                         exclude_cols=None):
    """Load and prepare test data."""
    if exclude_cols is None:
        exclude_cols = ["customer_id", "observation_date", target_col]
    
    df = pd.read_csv(features_file)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y, feature_cols


def main():
    """Main function to evaluate model."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model using standardized framework"
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to trained model pickle file",
    )
    parser.add_argument(
        "test_data",
        type=Path,
        help="Path to test features CSV file",
    )
    parser.add_argument(
        "--target-col",
        default="churn",
        help="Name of target column (default: churn)",
    )
    parser.add_argument(
        "--model-name",
        default="baseline",
        help="Name of the model (default: baseline)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for evaluation report (default: reports/{model_name}_metrics.json)",
    )
    parser.add_argument(
        "--include-curves",
        action="store_true",
        help="Include ROC and PR curve data in report",
    )
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = REPORTS_DIR / f"{args.model_name}_metrics.json"
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Output: {args.output}")
    print("-" * 60)
    
    # Load model
    print("\nLoading model...")
    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    model = load_model(args.model_path)
    print("Model loaded successfully")
    
    # Load test data
    print("\nLoading test data...")
    if not args.test_data.exists():
        print(f"Error: Test data file not found: {args.test_data}")
        sys.exit(1)
    
    X_test, y_test, feature_names = prepare_test_data(
        args.test_data, target_col=args.target_col
    )
    print(f"Test set: {len(X_test)} samples, {len(feature_names)} features")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_proba = None
    
    # Try to get probabilities if available
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except:
            pass
    
    # Evaluate
    print("\nEvaluating model...")
    report = evaluate_model(
        y_test, y_pred, y_proba, include_curves=args.include_curves
    )
    
    # Print summary
    print_metrics_summary(report["metrics"], set_name="Test")
    
    # Save report
    print("\nSaving evaluation report...")
    save_evaluation_report(
        report,
        args.output,
        model_name=args.model_name,
        dataset_name="test",
        additional_metadata={
            "n_features": len(feature_names),
            "n_samples": len(X_test),
            "features": feature_names,
        },
    )
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

