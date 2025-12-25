"""Command-line interface for the customer churn prediction pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from src.data.load_raw import main as load_raw_main
from src.data.sql_runner import main as sql_runner_main
from src.evaluation.analyze_thresholds import analyze_model_thresholds
from src.evaluation.generate_report import generate_baseline_report
from src.evaluation.run_business_evaluation import run_business_evaluation
from src.evaluation.run_error_analysis import run_error_analysis
from src.features.compute_features import main as compute_features_main
from src.features.validate_features import validate_features
from src.models.train_baseline import train_baseline
from src.models.train_tree import main as train_tree_main
from src.models.tune import main as tune_main
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_make_features(args):
    """Generate features from raw data."""
    logger.info("=" * 60)
    logger.info("Feature Generation Pipeline")
    logger.info("=" * 60)

    # Step 1: Load raw data
    logger.info("\nStep 1: Loading raw data...")
    try:
        load_raw_main(args.raw_file)
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        sys.exit(1)

    # Step 2: Build base tables (if needed)
    if args.build_tables:
        logger.info("\nStep 2: Building base tables...")
        try:
            sql_runner_main()
        except Exception as e:
            logger.error(f"Failed to build base tables: {e}")
            sys.exit(1)

    # Step 3: Compute features
    logger.info("\nStep 3: Computing features...")
    try:
        compute_features_main(
            db_path=args.db_path,
            sql_file=args.sql_file,
            output_path=args.output_path,
        )
    except Exception as e:
        logger.error(f"Failed to compute features: {e}")
        sys.exit(1)

    # Step 4: Validate features
    logger.info("\nStep 4: Validating features...")
    features_path = args.output_path or PROCESSED_DATA_DIR / "features.parquet"
    try:
        validate_features(features_path, strict=True)
    except Exception as e:
        logger.error(f"Feature validation failed: {e}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("Feature generation completed successfully")
    logger.info("=" * 60)


def cmd_train(args):
    """Train a model."""
    logger.info("=" * 60)
    logger.info(f"Training {args.model_type} Model")
    logger.info("=" * 60)

    if args.model_type == "baseline":
        try:
            train_baseline(
                features_path=args.features_path,
                model_path=args.model_path,
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)
    elif args.model_type in ["random_forest", "xgboost", "lightgbm"]:
        try:
            train_tree_main(
                model_type=args.model_type,
                features_path=args.features_path,
                model_path=args.model_path,
                report_path=args.report_path,
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("Model training completed successfully")
    logger.info("=" * 60)


def cmd_tune(args):
    """Tune hyperparameters."""
    logger.info("=" * 60)
    logger.info("Hyperparameter Tuning")
    logger.info("=" * 60)

    try:
        tune_main(
            model_type=args.model_type,
            features_path=args.features_path,
            n_iter=args.n_iter,
            cv=args.cv,
            scoring=args.scoring,
            model_path=args.model_path,
            summary_path=args.summary_path,
        )
    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("Hyperparameter tuning completed successfully")
    logger.info("=" * 60)


def cmd_evaluate(args):
    """Evaluate a trained model."""
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)

    # Generate metrics report
    if args.metrics:
        logger.info("\nGenerating metrics report...")
        try:
            from src.models.train_baseline import (
                load_features,
                prepare_features,
                temporal_split,
            )
            import pickle

            with open(args.model_path, "rb") as f:
                model = pickle.load(f)

            df = load_features(args.features_path)
            train_df, valid_df, test_df = temporal_split(df)

            X_train, y_train = prepare_features(train_df)
            X_valid, y_valid = prepare_features(valid_df)
            X_test, y_test = prepare_features(test_df)

            generate_baseline_report(
                model,
                X_train,
                y_train,
                X_valid,
                y_valid,
                X_test,
                y_test,
                output_path=args.report_path or REPORTS_DIR / "baseline_metrics.json",
            )
        except Exception as e:
            logger.error(f"Metrics generation failed: {e}")
            sys.exit(1)

    # Threshold analysis
    if args.thresholds:
        logger.info("\nPerforming threshold analysis...")
        try:
            analyze_model_thresholds(
                args.model_path,
                features_path=args.features_path,
                cost_fp=args.cost_fp,
                cost_fn=args.cost_fn,
                split=args.split,
            )
        except Exception as e:
            logger.error(f"Threshold analysis failed: {e}")
            sys.exit(1)

    # Business evaluation
    if args.business:
        logger.info("\nPerforming business evaluation...")
        try:
            run_business_evaluation(
                args.model_path,
                features_path=args.features_path,
                retention_cost=args.retention_cost,
                churn_loss=args.churn_loss,
                intervention_budget=args.budget,
                scenario_name=args.scenario_name,
                split=args.split,
            )
        except Exception as e:
            logger.error(f"Business evaluation failed: {e}")
            sys.exit(1)

    # Error analysis
    if args.errors:
        logger.info("\nPerforming error analysis...")
        try:
            run_error_analysis(
                args.model_path,
                features_path=args.features_path,
                split=args.split,
                fnr_threshold=args.fnr_threshold,
                threshold=args.threshold,
            )
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("Model evaluation completed successfully")
    logger.info("=" * 60)


def cmd_full_run(args):
    """Run the complete pipeline from raw data to evaluation."""
    logger.info("=" * 60)
    logger.info("Full Pipeline Execution")
    logger.info("=" * 60)

    # Step 1: Make features
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Feature Generation")
    logger.info("=" * 60)
    cmd_make_features(args)

    # Step 2: Train model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Model Training")
    logger.info("=" * 60)
    train_args = argparse.Namespace(
        model_type=args.model_type,
        features_path=args.features_path or PROCESSED_DATA_DIR / "features.parquet",
        model_path=args.model_path,
        report_path=None,
    )
    cmd_train(train_args)

    # Step 3: Evaluate model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Model Evaluation")
    logger.info("=" * 60)
    eval_args = argparse.Namespace(
        model_path=args.model_path or MODELS_DIR / f"{args.model_type}.pkl",
        features_path=args.features_path or PROCESSED_DATA_DIR / "features.parquet",
        metrics=True,
        thresholds=True,
        business=True,
        errors=True,
        report_path=None,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn,
        retention_cost=args.retention_cost,
        churn_loss=args.churn_loss,
        budget=args.budget,
        scenario_name=args.scenario_name,
        split=args.split,
        fnr_threshold=args.fnr_threshold,
        threshold=args.threshold,
    )
    cmd_evaluate(eval_args)

    logger.info("\n" + "=" * 60)
    logger.info("Full pipeline execution completed successfully")
    logger.info("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate features
  python -m src.cli make-features

  # Train baseline model
  python -m src.cli train --model-type baseline

  # Train random forest
  python -m src.cli train --model-type random_forest

  # Evaluate model
  python -m src.cli evaluate --model-path models/baseline.pkl --metrics --thresholds

  # Full pipeline
  python -m src.cli full-run --model-type baseline
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # make-features command
    parser_features = subparsers.add_parser("make-features", help="Generate features from raw data")
    parser_features.add_argument(
        "--raw-file",
        type=str,
        default=None,
        help="Path to raw data file (CSV/Parquet)",
    )
    parser_features.add_argument(
        "--build-tables",
        action="store_true",
        help="Build base SQL tables",
    )
    parser_features.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to DuckDB database",
    )
    parser_features.add_argument(
        "--sql-file",
        type=str,
        default=None,
        help="Path to SQL feature computation script",
    )
    parser_features.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save features Parquet file",
    )

    # train command
    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_train.add_argument(
        "--model-type",
        type=str,
        default="baseline",
        choices=["baseline", "random_forest", "xgboost", "lightgbm"],
        help="Type of model to train",
    )
    parser_train.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features Parquet file",
    )
    parser_train.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save trained model",
    )
    parser_train.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path to save evaluation report",
    )

    # tune command
    parser_tune = subparsers.add_parser("tune", help="Tune hyperparameters")
    parser_tune.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["logistic_regression", "random_forest"],
        help="Type of model to tune",
    )
    parser_tune.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features Parquet file",
    )
    parser_tune.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of hyperparameter configurations to try",
    )
    parser_tune.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser_tune.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Scoring metric for model selection",
    )
    parser_tune.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save tuned model",
    )
    parser_tune.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Path to save tuning summary",
    )

    # evaluate command
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    parser_eval.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser_eval.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features Parquet file",
    )
    parser_eval.add_argument(
        "--metrics",
        action="store_true",
        help="Generate metrics report",
    )
    parser_eval.add_argument(
        "--thresholds",
        action="store_true",
        help="Perform threshold analysis",
    )
    parser_eval.add_argument(
        "--business",
        action="store_true",
        help="Perform business evaluation",
    )
    parser_eval.add_argument(
        "--errors",
        action="store_true",
        help="Perform error analysis",
    )
    parser_eval.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path to save evaluation report",
    )
    parser_eval.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="Cost per false positive (for threshold analysis)",
    )
    parser_eval.add_argument(
        "--cost-fn",
        type=float,
        default=5.0,
        help="Cost per false negative (for threshold analysis)",
    )
    parser_eval.add_argument(
        "--retention-cost",
        type=float,
        default=10.0,
        help="Cost per retention intervention",
    )
    parser_eval.add_argument(
        "--churn-loss",
        type=float,
        default=100.0,
        help="Loss per churned customer",
    )
    parser_eval.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Intervention budget constraint",
    )
    parser_eval.add_argument(
        "--scenario-name",
        type=str,
        default="default",
        help="Business scenario name",
    )
    parser_eval.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Data split to evaluate",
    )
    parser_eval.add_argument(
        "--fnr-threshold",
        type=float,
        default=0.3,
        help="FNR threshold for error analysis",
    )
    parser_eval.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )

    # full-run command
    parser_full = subparsers.add_parser(
        "full-run", help="Run complete pipeline from raw data to evaluation"
    )
    parser_full.add_argument(
        "--raw-file",
        type=str,
        default=None,
        help="Path to raw data file",
    )
    parser_full.add_argument(
        "--model-type",
        type=str,
        default="baseline",
        choices=["baseline", "random_forest", "xgboost", "lightgbm"],
        help="Type of model to train",
    )
    parser_full.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features Parquet file",
    )
    parser_full.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save trained model",
    )
    parser_full.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="Cost per false positive",
    )
    parser_full.add_argument(
        "--cost-fn",
        type=float,
        default=5.0,
        help="Cost per false negative",
    )
    parser_full.add_argument(
        "--retention-cost",
        type=float,
        default=10.0,
        help="Cost per retention intervention",
    )
    parser_full.add_argument(
        "--churn-loss",
        type=float,
        default=100.0,
        help="Loss per churned customer",
    )
    parser_full.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Intervention budget constraint",
    )
    parser_full.add_argument(
        "--scenario-name",
        type=str,
        default="default",
        help="Business scenario name",
    )
    parser_full.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Data split to evaluate",
    )
    parser_full.add_argument(
        "--fnr-threshold",
        type=float,
        default=0.3,
        help="FNR threshold for error analysis",
    )
    parser_full.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "make-features":
        cmd_make_features(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "full-run":
        cmd_full_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

