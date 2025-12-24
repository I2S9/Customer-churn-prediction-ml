# Feature Specification

This document defines the features to be engineered for the customer churn prediction model.

## Churn Label Definition

### Definition
A customer is considered churned if they have no activity (no events) for a period of **N days** after the observation window.

### Temporal Window
- **Observation window**: Last 90 days of customer activity
- **Churn window**: 30 days after the observation window
- **Label calculation date**: End of observation window

### Label Logic
```
churn = 1 if (last_event_date < observation_end_date - churn_window_days)
churn = 0 otherwise
```

### Assumptions
- Customers with no events during the observation window are excluded (insufficient data)
- Customers who churned before the observation window are excluded (historical churn)

## Feature Categories

### 1. RFM Features (Recency, Frequency, Monetary)

#### Recency Features
- `days_since_last_event`: Number of days since the customer's last event
- `days_since_first_event`: Number of days since the customer's first event
- `days_since_signup`: Number of days since customer signup

#### Frequency Features
- `total_events`: Total number of events in observation window
- `unique_event_types`: Number of distinct event types
- `events_per_day`: Average events per day (total_events / observation_days)
- `active_days`: Number of distinct days with activity

#### Monetary/Usage Features
- `total_revenue`: Sum of all event values (if applicable)
- `avg_event_value`: Average value per event
- `max_event_value`: Maximum single event value
- `monthly_revenue`: Customer's monthly subscription revenue (from customers table)

### 2. Temporal Features

#### Rolling Window Aggregates
All calculated over the last 7, 14, and 30 days:
- `events_last_7d`: Event count in last 7 days
- `events_last_14d`: Event count in last 14 days
- `events_last_30d`: Event count in last 30 days
- `revenue_last_7d`: Revenue in last 7 days
- `revenue_last_14d`: Revenue in last 14 days
- `revenue_last_30d`: Revenue in last 30 days

#### Trend Features
- `event_trend_7d_30d`: Ratio of events in last 7 days vs last 30 days
- `event_trend_14d_30d`: Ratio of events in last 14 days vs last 30 days
- `revenue_trend_7d_30d`: Ratio of revenue in last 7 days vs last 30 days
- `revenue_trend_14d_30d`: Ratio of revenue in last 14 days vs last 30 days
- `activity_slope`: Linear trend of daily event counts over observation window

#### Time-based Patterns
- `weekend_activity_ratio`: Proportion of events on weekends
- `peak_hour_activity`: Events during peak hours (e.g., 9-17)
- `activity_consistency`: Coefficient of variation of daily event counts

### 3. Event Type Features

#### Per Event Type Aggregates
For each distinct event type:
- `count_{event_type}`: Number of events of this type
- `last_{event_type}_days_ago`: Days since last occurrence
- `avg_{event_type}_value`: Average value (if applicable)

### 4. Customer Profile Features

#### Demographics
- `country`: Customer country (encoded)
- `plan_type`: Subscription plan type (encoded)
- `customer_age_days`: Days since signup
- `plan_tier`: Plan tier category (if applicable)

#### Behavioral Segments
- `customer_segment`: RFM-based segment (e.g., "champion", "at_risk", "lost")
- `engagement_level`: Engagement level based on activity frequency

## Feature Engineering Rules

### Missing Values
- Missing event values: Set to 0
- Missing dates: Exclude customer from dataset
- Missing categorical: Use "unknown" category

### Feature Scaling
- Numerical features: Standard scaling (mean=0, std=1)
- Categorical features: One-hot encoding or target encoding

### Feature Selection
- Remove features with >90% constant values
- Remove features with correlation >0.95
- Use feature importance from baseline model for selection

## Implementation Notes

### Data Requirements
- Events table must have: `customer_id`, `event_timestamp`, `event_type`, `event_value`
- Customers table must have: `customer_id`, `signup_date`, `plan_type`, `monthly_revenue`

### Computation
- All features computed for a fixed observation date
- Features must be deterministic (same input = same output)
- Use SQL for aggregations when possible for performance

### Validation
- Feature distributions should be checked for outliers
- Feature-target relationships should be validated
- Features should be tested for data leakage (no future information)

## Out of Scope

The following features are explicitly **not** included in this specification:
- External data sources (weather, holidays, etc.)
- Social network features
- Text features from metadata
- Image or unstructured data features
- Real-time streaming features

This specification is frozen. Any additions require explicit approval and documentation update.

