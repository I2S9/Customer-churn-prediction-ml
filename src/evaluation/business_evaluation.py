"""Business-oriented evaluation with cost-benefit analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.evaluation.thresholds import analyze_thresholds, find_optimal_threshold
from src.utils.paths import REPORTS_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BusinessScenario:
    """Business scenario with costs and constraints."""

    def __init__(
        self,
        retention_cost_per_customer: float = 10.0,
        churn_loss_per_customer: float = 100.0,
        intervention_budget: Optional[float] = None,
        name: str = "default",
    ):
        """
        Initialize business scenario.

        Parameters
        ----------
        retention_cost_per_customer : float, default 10.0
            Cost of retention intervention per customer (e.g., discount, offer).
        churn_loss_per_customer : float, default 100.0
            Loss when a customer churns (e.g., LTV loss).
        intervention_budget : float, optional
            Maximum budget for interventions. If None, no budget constraint.
        name : str, default "default"
            Name of the scenario.
        """
        self.retention_cost = retention_cost_per_customer
        self.churn_loss = churn_loss_per_customer
        self.budget = intervention_budget
        self.name = name

    def calculate_net_gain(
        self,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
    ) -> float:
        """
        Calculate net gain for a given confusion matrix.

        Parameters
        ----------
        tp : int
            True positives (churned customers correctly identified).
        fp : int
            False positives (non-churned customers incorrectly targeted).
        fn : int
            False negatives (churned customers missed).
        tn : int
            True negatives (non-churned customers correctly ignored).

        Returns
        -------
        float
            Net gain: (saved churns * churn_loss) - (interventions * retention_cost)
        """
        # Gains: prevented churns (TP) save churn_loss each
        gains = tp * self.churn_loss

        # Costs: all interventions (TP + FP) cost retention_cost each
        intervention_cost = (tp + fp) * self.retention_cost

        # Losses: missed churns (FN) still result in churn_loss
        losses = fn * self.churn_loss

        # Net gain = gains - costs - losses
        # Note: losses are already accounted for in the baseline, so we calculate:
        # Net gain = (prevented losses) - (intervention costs)
        net_gain = gains - intervention_cost

        return float(net_gain)

    def calculate_roi(self, net_gain: float, total_interventions: int) -> float:
        """
        Calculate Return on Investment (ROI).

        Parameters
        ----------
        net_gain : float
            Net gain from interventions.
        total_interventions : int
            Total number of interventions (TP + FP).

        Returns
        -------
        float
            ROI as percentage: (net_gain / investment) * 100
        """
        if total_interventions == 0:
            return 0.0

        investment = total_interventions * self.retention_cost
        if investment == 0:
            return 0.0

        roi = (net_gain / investment) * 100
        return float(roi)


def evaluate_business_scenario(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    scenario: BusinessScenario,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluate business metrics across different thresholds.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        True binary labels.
    y_pred_proba : numpy.ndarray or pandas.Series
        Predicted probabilities for positive class.
    scenario : BusinessScenario
        Business scenario with costs and constraints.
    thresholds : numpy.ndarray, optional
        Thresholds to evaluate. If None, uses 100 evenly spaced thresholds.

    Returns
    -------
    pandas.DataFrame
        DataFrame with business metrics for each threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 100)

    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        total_interventions = tp + fp
        net_gain = scenario.calculate_net_gain(tp, fp, fn, tn)
        roi = scenario.calculate_roi(net_gain, total_interventions)

        # Check budget constraint
        total_cost = total_interventions * scenario.retention_cost
        within_budget = (
            scenario.budget is None or total_cost <= scenario.budget
        )

        results.append(
            {
                "threshold": float(threshold),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "total_interventions": int(total_interventions),
                "intervention_cost": float(total_cost),
                "prevented_churns": int(tp),
                "missed_churns": int(fn),
                "net_gain": float(net_gain),
                "roi_percent": float(roi),
                "within_budget": bool(within_budget),
            }
        )

    df = pd.DataFrame(results)
    return df


def find_best_threshold_under_budget(
    business_df: pd.DataFrame,
    scenario: BusinessScenario,
) -> Dict:
    """
    Find best threshold that respects budget constraint.

    Parameters
    ----------
    business_df : pandas.DataFrame
        DataFrame from evaluate_business_scenario.
    scenario : BusinessScenario
        Business scenario with budget constraint.

    Returns
    -------
    dict
        Dictionary with best threshold and metrics.
    """
    if scenario.budget is None:
        # No budget constraint, maximize net gain
        best_idx = business_df["net_gain"].idxmax()
    else:
        # Filter by budget constraint
        within_budget_df = business_df[business_df["within_budget"]]
        if len(within_budget_df) == 0:
            logger.warning(
                f"No threshold found within budget of {scenario.budget}. "
                "Returning threshold with minimum cost."
            )
            best_idx = business_df["intervention_cost"].idxmin()
        else:
            # Maximize net gain within budget
            best_idx = within_budget_df["net_gain"].idxmax()

    best_row = business_df.loc[best_idx]

    result = {
        "threshold": float(best_row["threshold"]),
        "net_gain": float(best_row["net_gain"]),
        "roi_percent": float(best_row["roi_percent"]),
        "total_interventions": int(best_row["total_interventions"]),
        "intervention_cost": float(best_row["intervention_cost"]),
        "prevented_churns": int(best_row["prevented_churns"]),
        "missed_churns": int(best_row["false_negatives"]),
        "true_positives": int(best_row["true_positives"]),
        "false_positives": int(best_row["false_positives"]),
        "false_negatives": int(best_row["false_negatives"]),
        "within_budget": bool(best_row["within_budget"]),
    }

    return result


def compare_with_baseline(
    baseline_net_gain: float,
    baseline_interventions: int,
    optimized_net_gain: float,
    optimized_interventions: int,
    scenario: BusinessScenario,
) -> Dict:
    """
    Compare optimized strategy with baseline.

    Parameters
    ----------
    baseline_net_gain : float
        Net gain from baseline strategy (e.g., no intervention or random).
    baseline_interventions : int
        Number of interventions in baseline strategy.
    optimized_net_gain : float
        Net gain from optimized strategy.
    optimized_interventions : int
        Number of interventions in optimized strategy.
    scenario : BusinessScenario
        Business scenario.

    Returns
    -------
    dict
        Comparison results.
    """
    incremental_gain = optimized_net_gain - baseline_net_gain
    incremental_investment = (
        optimized_interventions - baseline_interventions
    ) * scenario.retention_cost

    incremental_roi = (
        (incremental_gain / incremental_investment * 100)
        if incremental_investment > 0
        else 0.0
    )

    comparison = {
        "baseline": {
            "net_gain": float(baseline_net_gain),
            "interventions": int(baseline_interventions),
            "investment": float(baseline_interventions * scenario.retention_cost),
            "roi_percent": float(
                scenario.calculate_roi(baseline_net_gain, baseline_interventions)
            ),
        },
        "optimized": {
            "net_gain": float(optimized_net_gain),
            "interventions": int(optimized_interventions),
            "investment": float(optimized_interventions * scenario.retention_cost),
            "roi_percent": float(
                scenario.calculate_roi(optimized_net_gain, optimized_interventions)
            ),
        },
        "incremental": {
            "net_gain": float(incremental_gain),
            "interventions": int(optimized_interventions - baseline_interventions),
            "investment": float(incremental_investment),
            "roi_percent": float(incremental_roi),
        },
    }

    return comparison


def perform_business_evaluation(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    scenario: BusinessScenario,
    baseline_strategy: str = "no_intervention",
    output_path: Optional[str | Path] = None,
) -> Dict:
    """
    Perform comprehensive business evaluation.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        True binary labels.
    y_pred_proba : numpy.ndarray or pandas.Series
        Predicted probabilities for positive class.
    scenario : BusinessScenario
        Business scenario with costs and constraints.
    baseline_strategy : str, default "no_intervention"
        Baseline strategy: "no_intervention" or "random".
    output_path : str | Path, optional
        Path to save results JSON.

    Returns
    -------
    dict
        Complete business evaluation results.
    """
    logger.info("=" * 60)
    logger.info(f"Business Evaluation: {scenario.name}")
    logger.info("=" * 60)
    logger.info(f"Retention cost: ${scenario.retention_cost:.2f} per customer")
    logger.info(f"Churn loss: ${scenario.churn_loss:.2f} per customer")
    if scenario.budget:
        logger.info(f"Budget constraint: ${scenario.budget:.2f}")

    # Evaluate business scenario
    logger.info("Evaluating business metrics across thresholds...")
    business_df = evaluate_business_scenario(y_true, y_pred_proba, scenario)

    # Find best threshold under budget
    logger.info("Finding best threshold under budget constraint...")
    best_threshold_result = find_best_threshold_under_budget(business_df, scenario)

    # Calculate baseline
    logger.info(f"Calculating baseline ({baseline_strategy})...")
    if baseline_strategy == "no_intervention":
        baseline_net_gain = 0.0
        baseline_interventions = 0
    elif baseline_strategy == "random":
        # Random intervention on 50% of customers
        n_customers = len(y_true)
        baseline_interventions = n_customers // 2
        baseline_tp = (y_true == 1).sum() // 2  # Rough estimate
        baseline_fp = baseline_interventions - baseline_tp
        baseline_fn = (y_true == 1).sum() - baseline_tp
        baseline_net_gain = scenario.calculate_net_gain(
            baseline_tp, baseline_fp, baseline_fn, n_customers - baseline_interventions
        )
    else:
        raise ValueError(f"Unknown baseline_strategy: {baseline_strategy}")

    # Compare with baseline
    comparison = compare_with_baseline(
        baseline_net_gain,
        baseline_interventions,
        best_threshold_result["net_gain"],
        best_threshold_result["total_interventions"],
        scenario,
    )

    # Compile results
    results = {
        "scenario": {
            "name": scenario.name,
            "retention_cost_per_customer": scenario.retention_cost,
            "churn_loss_per_customer": scenario.churn_loss,
            "intervention_budget": scenario.budget,
        },
        "best_threshold": best_threshold_result,
        "baseline_strategy": baseline_strategy,
        "comparison": comparison,
        "summary": {
            "incremental_net_gain": comparison["incremental"]["net_gain"],
            "incremental_roi": comparison["incremental"]["roi_percent"],
            "total_customers": int(len(y_true)),
            "total_churned": int((y_true == 1).sum()),
            "churn_rate": float((y_true == 1).mean()),
        },
    }

    # Save results
    if output_path is None:
        output_path = REPORTS_DIR / f"business_evaluation_{scenario.name}.json"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    logger.info(f"Saving results to: {output_path}")

    def convert_to_json_serializable(obj):
        """Convert numpy types to native Python types."""
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

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("Business Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Best threshold: {best_threshold_result['threshold']:.4f}")
    logger.info(f"Net gain: ${best_threshold_result['net_gain']:,.2f}")
    logger.info(f"ROI: {best_threshold_result['roi_percent']:.2f}%")
    logger.info(f"Interventions: {best_threshold_result['total_interventions']:,}")
    logger.info(f"Prevented churns: {best_threshold_result['prevented_churns']:,}")
    logger.info("\nComparison with Baseline:")
    logger.info(
        f"  Incremental net gain: ${comparison['incremental']['net_gain']:,.2f}"
    )
    logger.info(
        f"  Incremental ROI: {comparison['incremental']['roi_percent']:.2f}%"
    )
    logger.info("=" * 60)

    return results


def create_comparison_table(
    results: Dict,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Create comparison table for business evaluation.

    Parameters
    ----------
    results : dict
        Results from perform_business_evaluation.
    output_path : str | Path, optional
        Path to save CSV table.

    Returns
    -------
    pandas.DataFrame
        Comparison table.
    """
    comparison = results["comparison"]

    table_data = [
        {
            "Strategy": "Baseline",
            "Net Gain ($)": comparison["baseline"]["net_gain"],
            "Interventions": comparison["baseline"]["interventions"],
            "Investment ($)": comparison["baseline"]["investment"],
            "ROI (%)": comparison["baseline"]["roi_percent"],
        },
        {
            "Strategy": "Optimized",
            "Net Gain ($)": comparison["optimized"]["net_gain"],
            "Interventions": comparison["optimized"]["interventions"],
            "Investment ($)": comparison["optimized"]["investment"],
            "ROI (%)": comparison["optimized"]["roi_percent"],
        },
        {
            "Strategy": "Incremental",
            "Net Gain ($)": comparison["incremental"]["net_gain"],
            "Interventions": comparison["incremental"]["interventions"],
            "Investment ($)": comparison["incremental"]["investment"],
            "ROI (%)": comparison["incremental"]["roi_percent"],
        },
    ]

    df = pd.DataFrame(table_data)

    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        df.to_csv(output_path, index=False)
        logger.info(f"Comparison table saved to: {output_path}")

    return df


def main(
    y_true_path: Optional[str] = None,
    y_pred_proba_path: Optional[str] = None,
    retention_cost: float = 10.0,
    churn_loss: float = 100.0,
    budget: Optional[float] = None,
    scenario_name: str = "default",
) -> None:
    """
    Main function for business evaluation.

    Parameters
    ----------
    y_true_path : str, optional
        Path to CSV/Parquet with true labels.
    y_pred_proba_path : str, optional
        Path to CSV/Parquet with predicted probabilities.
    retention_cost : float, default 10.0
        Cost per retention intervention.
    churn_loss : float, default 100.0
        Loss per churned customer.
    budget : float, optional
        Budget constraint for interventions.
    scenario_name : str, default "default"
        Name of the scenario.
    """
    logger.info("Business evaluation module loaded. Use perform_business_evaluation() function.")


if __name__ == "__main__":
    logger.info("Business evaluation module loaded. Use perform_business_evaluation() function.")

