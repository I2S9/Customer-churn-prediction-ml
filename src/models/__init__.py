"""Model definitions and training logic."""

from src.models.train_baseline import (
    create_baseline_pipeline,
    evaluate_model,
    load_features,
    main as baseline_main,
    prepare_features,
    save_model,
    temporal_split,
    train_baseline,
)
from src.models.train_tree import (
    compare_models,
    create_lightgbm_pipeline,
    create_random_forest_pipeline,
    create_xgboost_pipeline,
    generate_tree_report,
    main as tree_main,
    train_tree_model,
)
from src.models.tune import (
    create_pipeline,
    get_param_grid,
    main as tune_main,
    save_tuned_model,
    save_tuning_summary,
    tune_model,
)

__all__ = [
    "baseline_main",
    "compare_models",
    "create_baseline_pipeline",
    "create_lightgbm_pipeline",
    "create_pipeline",
    "create_random_forest_pipeline",
    "create_xgboost_pipeline",
    "evaluate_model",
    "generate_tree_report",
    "get_param_grid",
    "load_features",
    "prepare_features",
    "save_model",
    "save_tuned_model",
    "save_tuning_summary",
    "temporal_split",
    "train_baseline",
    "train_tree_model",
    "tree_main",
    "tune_main",
    "tune_model",
]

