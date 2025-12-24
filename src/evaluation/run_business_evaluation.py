"""Run business evaluation on trained models."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.business_evaluation import (
    BusinessScenario,
    create_comparison_table,
    perform_business_evaluation,
)
from src.models.train_baseline import load_features, prepare_features, temporal_split
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_business_evaluation(
    model_path: str | Path,
    features_path: Optional[str | Path] = None,
    retention_cost: float = 10.0,
    churn_loss: float = 100.0,
    intervention_budget: Optional[float] = None,
    scenario_name: str = "default",
    split: str = "test",
    baseline_strategy: str = "no_intervention",
) -> dict:
    """
    Run business evaluation on a trained model.

    Parameters
    ----------
    model_path : str | Path
        Path to saved model pickle file.
    features_path : str | Path, optional
        Path to features Parquet file.
    retention_cost : float, default 10.0
        Cost per retention intervention.
    churn_loss : float, default 100.0
        Loss per churned customer.
    intervention_budget : float, optional
        Budget constraint for interventions.
    scenario_name : str, default "default"
        Name of the business scenario.
    split : str, default "test"
        Data split to evaluate ("train", "validation", or "test").
    baseline_strategy : str, default "no_intervention"
        Baseline strategy for comparison.

    Returns
    -------
    dict
        Business evaluation results.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if features_path is None:
        features_path = PROCESSED_DATA_DIR / "features.parquet"
    else:
        features_path = Path(features_path)

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

    # Create business scenario
    scenario = BusinessScenario(
        retention_cost_per_customer=retention_cost,
        churn_loss_per_customer=churn_loss,
        intervention_budget=intervention_budget,
        name=scenario_name,
    )

    # Perform business evaluation
    results = perform_business_evaluation(
        y_split,
        y_pred_proba,
        scenario,
        baseline_strategy=baseline_strategy,
    )

    # Create comparison table
    table_df = create_comparison_table(
        results, REPORTS_DIR / f"business_comparison_{scenario_name}.csv"
    )

    logger.info("\nComparison Table:")
    logger.info("\n" + table_df.to_string(index=False))

    return results


def main(
    model_path: Optional[str] = None,
    features_path: Optional[str] = None,
    retention_cost: float = 10.0,
    churn_loss: float = 100.0,
    budget: Optional[float] = None,
    scenario_name: str = "default",
    split: str = "test",
) -> None:
    """
    Main function to run business evaluation.

    Parameters
    ----------
    model_path : str, optional
        Path to saved model. Defaults to models/baseline.pkl.
    features_path : str, optional
        Path to features Parquet file.
    retention_cost : float, default 10.0
        Cost per retention intervention.
    churn_loss : float, default 100.0
        Loss per churned customer.
    budget : float, optional
        Budget constraint.
    scenario_name : str, default "default"
        Scenario name.
    split : str, default "test"
        Data split to evaluate.
    """
    if model_path is None:
        model_path = MODELS_DIR / "baseline.pkl"

    run_business_evaluation(
        model_path=model_path,
        features_path=features_path,
        retention_cost=retention_cost,
        churn_loss=churn_loss,
        intervention_budget=budget,
        scenario_name=scenario_name,
        split=split,
    )


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    features_path = sys.argv[2] if len(sys.argv) > 2 else None
    retention_cost = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    churn_loss = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
    budget = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "None" else None
    scenario_name = sys.argv[6] if len(sys.argv) > 6 else "default"
    split = sys.argv[7] if len(sys.argv) > 7 else "test"

    main(
        model_path=model_path,
        features_path=features_path,
        retention_cost=retention_cost,
        churn_loss=churn_loss,
        budget=budget,
        scenario_name=scenario_name,
        split=split,
    )

