-- Compute features for customer churn prediction
-- This script aggregates customer data and events to create feature vectors
-- Observation window: 90 days, Churn window: 30 days

-- Create or replace features table
CREATE OR REPLACE TABLE features AS
WITH observation_params AS (
    -- Set observation date (can be parameterized)
    SELECT 
        CURRENT_DATE AS observation_date,
        90 AS observation_days,
        30 AS churn_window_days
),
customer_events AS (
    -- Get all events within observation window
    SELECT 
        e.customer_id,
        e.event_timestamp,
        e.event_type,
        COALESCE(e.event_value, 0) AS event_value,
        DATE(e.event_timestamp) AS event_date,
        EXTRACT(hour FROM e.event_timestamp) AS event_hour,
        EXTRACT(dow FROM e.event_timestamp) AS day_of_week
    FROM events e
    CROSS JOIN observation_params op
    WHERE DATE(e.event_timestamp) >= DATE(op.observation_date) - op.observation_days
        AND DATE(e.event_timestamp) < DATE(op.observation_date)
),
customer_aggregates AS (
    -- RFM and temporal aggregates
    SELECT 
        ce.customer_id,
        -- Recency features
        (SELECT observation_date FROM observation_params) - MAX(ce.event_timestamp)::DATE AS days_since_last_event,
        (SELECT observation_date FROM observation_params) - MIN(ce.event_timestamp)::DATE AS days_since_first_event,
        
        -- Frequency features
        COUNT(*) AS total_events,
        COUNT(DISTINCT ce.event_type) AS unique_event_types,
        COUNT(*) * 1.0 / (SELECT observation_days FROM observation_params) AS events_per_day,
        COUNT(DISTINCT ce.event_date) AS active_days,
        
        -- Monetary features
        SUM(ce.event_value) AS total_revenue,
        AVG(ce.event_value) AS avg_event_value,
        MAX(ce.event_value) AS max_event_value,
        
        -- Rolling window aggregates (7, 14, 30 days)
        SUM(CASE WHEN ce.event_timestamp >= (SELECT observation_date FROM observation_params) - INTERVAL 7 DAY THEN 1 ELSE 0 END) AS events_last_7d,
        SUM(CASE WHEN ce.event_timestamp >= (SELECT observation_date FROM observation_params) - INTERVAL 14 DAY THEN 1 ELSE 0 END) AS events_last_14d,
        SUM(CASE WHEN ce.event_timestamp >= (SELECT observation_date FROM observation_params) - INTERVAL 30 DAY THEN 1 ELSE 0 END) AS events_last_30d,
        SUM(CASE WHEN ce.event_timestamp >= (SELECT observation_date FROM observation_params) - INTERVAL 7 DAY THEN ce.event_value ELSE 0 END) AS revenue_last_7d,
        SUM(CASE WHEN ce.event_timestamp >= (SELECT observation_date FROM observation_params) - INTERVAL 14 DAY THEN ce.event_value ELSE 0 END) AS revenue_last_14d,
        SUM(CASE WHEN ce.event_timestamp >= (SELECT observation_date FROM observation_params) - INTERVAL 30 DAY THEN ce.event_value ELSE 0 END) AS revenue_last_30d,
        
        -- Time-based patterns
        SUM(CASE WHEN ce.day_of_week IN (0, 6) THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS weekend_activity_ratio,
        SUM(CASE WHEN ce.event_hour BETWEEN 9 AND 17 THEN 1 ELSE 0 END) AS peak_hour_activity,
        
        -- Last event timestamp for churn calculation
        MAX(ce.event_timestamp) AS last_event_timestamp
    FROM customer_events ce
    GROUP BY ce.customer_id
),
daily_activity AS (
    -- Daily event counts for trend calculation
    SELECT 
        customer_id,
        event_date,
        COUNT(*) AS daily_events
    FROM customer_events
    GROUP BY customer_id, event_date
),
activity_trends AS (
    -- Calculate activity slope (linear trend)
    SELECT 
        customer_id,
        CASE 
            WHEN COUNT(*) > 1 THEN
                (COUNT(*) * SUM(EXTRACT(epoch FROM (event_date - MIN(event_date) OVER (PARTITION BY customer_id)))) - 
                 SUM(EXTRACT(epoch FROM (event_date - MIN(event_date) OVER (PARTITION BY customer_id)))) * SUM(daily_events)) * 1.0 /
                NULLIF((COUNT(*) * SUM(POWER(EXTRACT(epoch FROM (event_date - MIN(event_date) OVER (PARTITION BY customer_id))), 2)) - 
                        POWER(SUM(EXTRACT(epoch FROM (event_date - MIN(event_date) OVER (PARTITION BY customer_id)))), 2)), 0)
            ELSE 0
        END AS activity_slope,
        STDDEV(daily_events) * 1.0 / NULLIF(AVG(daily_events), 0) AS activity_consistency
    FROM daily_activity
    GROUP BY customer_id
),
event_type_features AS (
    -- Per event type aggregates (pivot-like)
    SELECT 
        customer_id,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS count_purchase,
        SUM(CASE WHEN event_type = 'purchase' THEN event_value ELSE 0 END) / NULLIF(SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END), 0) AS avg_purchase_value,
        (SELECT observation_date FROM observation_params) - MAX(CASE WHEN event_type = 'purchase' THEN event_timestamp END)::DATE AS last_purchase_days_ago,
        
        SUM(CASE WHEN event_type = 'login' THEN 1 ELSE 0 END) AS count_login,
        (SELECT observation_date FROM observation_params) - MAX(CASE WHEN event_type = 'login' THEN event_timestamp END)::DATE AS last_login_days_ago,
        
        SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS count_page_view,
        (SELECT observation_date FROM observation_params) - MAX(CASE WHEN event_type = 'page_view' THEN event_timestamp END)::DATE AS last_page_view_days_ago
    FROM customer_events
    GROUP BY customer_id
),
churn_labels AS (
    -- Calculate churn label
    SELECT 
        ca.customer_id,
        CASE 
            WHEN ca.last_event_timestamp < (SELECT observation_date FROM observation_params) - (SELECT churn_window_days FROM observation_params)
                THEN 1
            ELSE 0
        END AS churn
    FROM customer_aggregates ca
    CROSS JOIN observation_params op
    WHERE ca.total_events > 0  -- Exclude customers with no activity
)
SELECT 
    c.customer_id,
    cl.churn AS label,
    
    -- Customer profile features
    COALESCE(c.country, 'unknown') AS country,
    COALESCE(c.plan_type, 'unknown') AS plan_type,
    (SELECT observation_date FROM observation_params) - c.signup_date AS customer_age_days,
    COALESCE(c.monthly_revenue, 0) AS monthly_revenue,
    
    -- RFM features
    COALESCE(ca.days_since_last_event, 999) AS days_since_last_event,
    COALESCE(ca.days_since_first_event, 0) AS days_since_first_event,
    COALESCE(ca.total_events, 0) AS total_events,
    COALESCE(ca.unique_event_types, 0) AS unique_event_types,
    COALESCE(ca.events_per_day, 0) AS events_per_day,
    COALESCE(ca.active_days, 0) AS active_days,
    COALESCE(ca.total_revenue, 0) AS total_revenue,
    COALESCE(ca.avg_event_value, 0) AS avg_event_value,
    COALESCE(ca.max_event_value, 0) AS max_event_value,
    
    -- Rolling window features
    COALESCE(ca.events_last_7d, 0) AS events_last_7d,
    COALESCE(ca.events_last_14d, 0) AS events_last_14d,
    COALESCE(ca.events_last_30d, 0) AS events_last_30d,
    COALESCE(ca.revenue_last_7d, 0) AS revenue_last_7d,
    COALESCE(ca.revenue_last_14d, 0) AS revenue_last_14d,
    COALESCE(ca.revenue_last_30d, 0) AS revenue_last_30d,
    
    -- Trend features
    CASE 
        WHEN ca.events_last_30d > 0 THEN ca.events_last_7d * 1.0 / ca.events_last_30d
        ELSE 0
    END AS event_trend_7d_30d,
    CASE 
        WHEN ca.events_last_30d > 0 THEN ca.events_last_14d * 1.0 / ca.events_last_30d
        ELSE 0
    END AS event_trend_14d_30d,
    CASE 
        WHEN ca.revenue_last_30d > 0 THEN ca.revenue_last_7d * 1.0 / ca.revenue_last_30d
        ELSE 0
    END AS revenue_trend_7d_30d,
    CASE 
        WHEN ca.revenue_last_30d > 0 THEN ca.revenue_last_14d * 1.0 / ca.revenue_last_30d
        ELSE 0
    END AS revenue_trend_14d_30d,
    COALESCE(at.activity_slope, 0) AS activity_slope,
    
    -- Time-based patterns
    COALESCE(ca.weekend_activity_ratio, 0) AS weekend_activity_ratio,
    COALESCE(ca.peak_hour_activity, 0) AS peak_hour_activity,
    COALESCE(at.activity_consistency, 0) AS activity_consistency,
    
    -- Event type features
    COALESCE(etf.count_purchase, 0) AS count_purchase,
    COALESCE(etf.avg_purchase_value, 0) AS avg_purchase_value,
    COALESCE(etf.last_purchase_days_ago, 999) AS last_purchase_days_ago,
    COALESCE(etf.count_login, 0) AS count_login,
    COALESCE(etf.last_login_days_ago, 999) AS last_login_days_ago,
    COALESCE(etf.count_page_view, 0) AS count_page_view,
    COALESCE(etf.last_page_view_days_ago, 999) AS last_page_view_days_ago
    
FROM customers c
LEFT JOIN churn_labels cl ON c.customer_id = cl.customer_id
LEFT JOIN customer_aggregates ca ON c.customer_id = ca.customer_id
LEFT JOIN activity_trends at ON c.customer_id = at.customer_id
LEFT JOIN event_type_features etf ON c.customer_id = etf.customer_id
WHERE cl.customer_id IS NOT NULL  -- Only include customers with activity in observation window
;

-- Create index on customer_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_features_customer_id ON features(customer_id);

