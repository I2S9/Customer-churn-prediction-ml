# Customer Churn Prediction

> End-to-end machine learning pipeline for predicting customer churn using historical behavioral data. This project demonstrates professional data science practices with a focus on scalability, business metrics, and actionable insights rather than academic experimentation.

## Problem Statement

Customer churn represents a significant business challenge. This project predicts which customers are likely to churn within a 30-day window after a 90-day observation period, enabling proactive retention interventions. The solution emphasizes decision-making under business constraints, with cost-benefit analysis and threshold optimization to maximize return on investment.

## Data Description

The pipeline processes customer behavioral data including:
- **Customer profiles**: signup date, country, plan type, monthly revenue
- **Event data**: timestamped customer interactions (purchases, logins, page views) with values
- **Features**: RFM metrics (Recency, Frequency, Monetary), temporal patterns, and event-type aggregates

Data flows through: `raw data` → `staging` → `feature engineering (SQL)` → `modeling` → `evaluation`

## Setup

### Prerequisites

- Python >= 3.8
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Customer-churn-prediction-ml

# Install the package and development dependencies
pip install -e ".[dev]"
```

## Run Pipeline

### Quick Start

Run the complete pipeline from raw data to evaluation:

```bash
python -m src.cli full-run --model-type baseline
```

### Step-by-Step

```bash
# 1. Generate features from raw data
python -m src.cli make-features --raw-file data/raw/your_data.csv

# 2. Train a baseline model
python -m src.cli train --model-type baseline

# 3. Train a tree-based model (RandomForest)
python -m src.cli train --model-type random_forest

# 4. Evaluate model (metrics, thresholds, business, errors)
python -m src.cli evaluate \
    --model-path models/baseline.pkl \
    --metrics --thresholds --business --errors

# 5. Hyperparameter tuning
python -m src.cli tune --model-type random_forest --n-iter 30
```

### CLI Commands

- `make-features`: Load raw data, compute features via SQL, validate
- `train`: Train baseline or tree-based models (RandomForest/XGBoost/LightGBM)
- `tune`: Hyperparameter tuning with RandomizedSearchCV (controlled budget)
- `evaluate`: Comprehensive evaluation (metrics, thresholds, business, errors)
- `full-run`: Execute complete pipeline end-to-end

## Metrics

The pipeline evaluates models using multiple metrics:

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Ranking**: ROC-AUC, PR-AUC
- **Business**: Net Gain, ROI, Cost-Benefit Analysis
- **Error Analysis**: Segment-level FNR/FPR identification

Results are saved in `reports/`:
- `baseline_metrics.json` / `random_forest_metrics.json`: Model performance
- `threshold_analysis.json`: Optimal threshold under cost constraints
- `business_evaluation_*.json`: Business impact analysis
- `error_analysis.md`: Failure mode identification by segment

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw input data (never modified)
│   └── processed/        # Cleaned and feature-engineered datasets
├── sql/                  # SQL queries for feature computation
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/          # Model definitions and training
│   ├── evaluation/      # Metrics, validation, threshold analysis
│   ├── utils/           # Shared utilities (config, paths)
│   └── cli.py           # Command-line interface
├── notebooks/           # Exploratory analysis and visualization
├── reports/             # Generated figures and evaluation results
├── tests/               # Unit tests
└── docs/                # Documentation (feature specifications)
```

## Key Features

- **SQL-based feature engineering**: Scalable feature computation using DuckDB
- **Temporal validation**: Time-aware train/validation/test splits
- **Business-oriented evaluation**: Cost-benefit analysis with ROI calculation
- **Threshold optimization**: Find optimal decision threshold under budget constraints
- **Error analysis**: Identify failure modes by customer segment
- **Reproducible**: Fixed seeds, deterministic feature engineering, versioned dependencies

## Results Summary

View comprehensive results in the summary notebook:

```bash
jupyter notebook notebooks/01_results_summary.ipynb
```

## Decision Framing

The pipeline supports decision-making under business constraints:

### Threshold Selection

Classification threshold is optimized based on business costs:
- **False Positive Cost**: Cost of unnecessary retention intervention
- **False Negative Cost**: Cost of missed churn (lost customer value)
- **Optimal Threshold**: Minimizes total cost: `cost_fp × FP + cost_fn × FN`

### Budget Constraints

The system can find the best threshold under budget limitations:
- Maximum intervention budget
- ROI calculation for each strategy
- Comparison with baseline (no intervention)

### Business Impact

Results include:
- Net gain estimation vs baseline
- ROI calculation
- Intervention volume and cost
- Prevented churns

## Limitations

### Data Bias

- **Temporal bias**: Model trained on historical data may not generalize to future patterns
- **Selection bias**: Only customers with sufficient activity are included
- **Label bias**: Churn definition (30-day inactivity) may not capture all churn types

### Label Definition

- Churn is defined as 30 days of inactivity after 90-day observation window
- Customers with no events during observation are excluded
- Historical churns (before observation window) are excluded
- This definition may not align with all business contexts

### Leakage Risks

- Features are computed only from data before the observation date
- Manual review recommended to ensure no future information leakage
- Temporal ordering is enforced in train/validation/test splits
- Feature validation checks for suspicious correlations

### Model Assumptions

- Stationarity assumption: customer behavior patterns remain stable
- Linearity assumption (baseline model): Logistic regression assumes linear relationships
- Independence assumption: Customer behaviors are treated as independent

## Next Steps

### Production Deployment

1. **Online A/B Testing**: Deploy model to production with controlled experiments
   - Test intervention strategies on random customer subsets
   - Measure actual retention rates vs predicted
   - Compare treatment vs control groups

2. **Model Monitoring**: Set up continuous monitoring
   - Track prediction distributions over time
   - Monitor feature drift
   - Alert on performance degradation

3. **Feedback Loop**: Incorporate production feedback
   - Collect actual churn outcomes
   - Retrain model with new data periodically
   - Update feature engineering based on new patterns

### Model Improvements

- **Feature Engineering**: Add domain-specific features based on error analysis
- **Ensemble Methods**: Combine multiple models for robustness
- **Online Learning**: Adapt model to changing customer behavior patterns
- **Causal Inference**: Understand intervention effectiveness beyond correlation

### Business Integration

- **Real-time Scoring**: Deploy model for real-time churn prediction
- **Intervention Automation**: Automate retention campaigns based on predictions
- **ROI Tracking**: Measure actual ROI of interventions in production

## API Inference (Optional)

A simple FastAPI service is available for real-time predictions:

```bash
# Start the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# API documentation available at http://localhost:8000/docs
```

The API provides:
- `/predict`: Single customer prediction with recommended action
- `/predict-batch`: Batch predictions for multiple customers
- Automatic loading of optimal threshold and business parameters

See `src/api/README.md` for detailed usage.

## Testing

Run tests to verify pipeline components:

```bash
pytest tests/ -v
```