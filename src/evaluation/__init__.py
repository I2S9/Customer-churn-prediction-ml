"""Metrics, validation, and threshold analysis."""

from src.evaluation.analyze_thresholds import (
    analyze_model_thresholds,
    main as analyze_thresholds_main,
)
from src.evaluation.business_evaluation import (
    BusinessScenario,
    compare_with_baseline,
    create_comparison_table,
    evaluate_business_scenario,
    find_best_threshold_under_budget,
    perform_business_evaluation,
)
from src.evaluation.error_analysis import (
    analyze_errors_by_segment,
    create_segments,
    generate_error_report,
    identify_failure_segments,
    perform_error_analysis,
    plot_error_analysis,
)
from src.evaluation.generate_report import generate_baseline_report, main as generate_main
from src.evaluation.metrics import (
    calculate_metrics,
    compare_metrics,
    evaluate_model,
    load_metrics,
    main,
    save_metrics,
)
from src.evaluation.run_business_evaluation import (
    main as run_business_eval_main,
    run_business_evaluation,
)
from src.evaluation.run_error_analysis import (
    main as run_error_analysis_main,
    run_error_analysis,
)
from src.evaluation.thresholds import (
    analyze_thresholds,
    calculate_cost,
    find_optimal_threshold,
    perform_threshold_analysis,
    plot_cost_vs_threshold,
    plot_precision_recall_vs_threshold,
    plot_top_k_analysis,
    top_k_targeting,
)

__all__ = [
    "analyze_errors_by_segment",
    "analyze_model_thresholds",
    "analyze_thresholds",
    "analyze_thresholds_main",
    "BusinessScenario",
    "calculate_cost",
    "calculate_metrics",
    "compare_metrics",
    "compare_with_baseline",
    "create_comparison_table",
    "create_segments",
    "evaluate_business_scenario",
    "evaluate_model",
    "find_best_threshold_under_budget",
    "find_optimal_threshold",
    "generate_baseline_report",
    "generate_error_report",
    "generate_main",
    "identify_failure_segments",
    "load_metrics",
    "main",
    "perform_business_evaluation",
    "perform_error_analysis",
    "perform_threshold_analysis",
    "plot_cost_vs_threshold",
    "plot_error_analysis",
    "plot_precision_recall_vs_threshold",
    "plot_top_k_analysis",
    "run_business_evaluation",
    "run_business_eval_main",
    "run_error_analysis",
    "run_error_analysis_main",
    "save_metrics",
    "top_k_targeting",
]

