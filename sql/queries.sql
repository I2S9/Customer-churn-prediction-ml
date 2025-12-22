-- Customer Churn Prediction - Core Workload Queries
-- Version 1.0
--
-- This file contains representative queries for customer churn analysis.
-- Queries include: JOIN, aggregation, GROUP BY, subqueries, and selective filters.
-- All queries are designed to execute on the small dataset.

-- Query 1: Orders count by country
-- JOIN + GROUP BY + aggregation
SELECT 
    c.country, 
    COUNT(*) AS orders_count
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.country
ORDER BY orders_count DESC;

-- Query 2: Top customers by total revenue
-- JOIN + aggregation + GROUP BY + ORDER BY
SELECT 
    c.customer_id,
    c.country,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_revenue
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.country
ORDER BY total_revenue DESC
LIMIT 10;

-- Query 3: Most sold products by quantity
-- JOIN + aggregation + GROUP BY
SELECT 
    p.product_name,
    p.category,
    SUM(oi.quantity) AS total_quantity_sold,
    SUM(oi.quantity * oi.unit_price) AS total_revenue
FROM products p
JOIN order_items oi ON oi.product_id = p.product_id
JOIN orders o ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_quantity_sold DESC
LIMIT 10;

-- Query 4: Customers with no orders (potential churn candidates)
-- Subquery + filter
SELECT 
    c.customer_id,
    c.country,
    c.created_at,
    c.registration_channel
FROM customers c
WHERE c.customer_id NOT IN (
    SELECT DISTINCT customer_id 
    FROM orders
)
ORDER BY c.created_at DESC;

-- Query 5: Average order value by country
-- JOIN + aggregation + GROUP BY
SELECT 
    c.country,
    COUNT(DISTINCT o.order_id) AS order_count,
    AVG(o.total_amount) AS avg_order_value,
    MIN(o.total_amount) AS min_order_value,
    MAX(o.total_amount) AS max_order_value
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.country
ORDER BY avg_order_value DESC;

-- Query 6: Customer interactions by type and outcome
-- GROUP BY + aggregation + filter
SELECT 
    interaction_type,
    channel,
    outcome,
    COUNT(*) AS interaction_count
FROM customer_interactions
WHERE interaction_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY interaction_type, channel, outcome
ORDER BY interaction_count DESC;

-- Query 7: Revenue by product category
-- Multi-JOIN + GROUP BY + aggregation
SELECT 
    p.category,
    COUNT(DISTINCT o.order_id) AS orders_count,
    COUNT(oi.order_item_id) AS items_count,
    SUM(oi.quantity * oi.unit_price) AS total_revenue,
    AVG(oi.unit_price) AS avg_product_price
FROM products p
JOIN order_items oi ON oi.product_id = p.product_id
JOIN orders o ON o.order_id = oi.order_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY total_revenue DESC;

-- Query 8: Customers with high interaction count (support issues)
-- JOIN + aggregation + GROUP BY + filter
SELECT 
    c.customer_id,
    c.country,
    COUNT(ci.interaction_id) AS interaction_count,
    COUNT(CASE WHEN ci.outcome = 'escalated' THEN 1 END) AS escalated_count
FROM customers c
JOIN customer_interactions ci ON ci.customer_id = c.customer_id
GROUP BY c.customer_id, c.country
HAVING COUNT(ci.interaction_id) >= 3
ORDER BY interaction_count DESC;

-- Query 9: Monthly order trends
-- GROUP BY date + aggregation
SELECT 
    DATE_TRUNC('month', o.order_date) AS order_month,
    COUNT(DISTINCT o.order_id) AS orders_count,
    COUNT(DISTINCT o.customer_id) AS unique_customers,
    SUM(o.total_amount) AS total_revenue
FROM orders o
WHERE o.status = 'completed'
GROUP BY DATE_TRUNC('month', o.order_date)
ORDER BY order_month DESC;

-- Query 10: Customers with declining order frequency
-- Subquery + JOIN + filter
SELECT 
    c.customer_id,
    c.country,
    recent_orders.order_count AS recent_orders,
    all_orders.order_count AS total_orders,
    ROUND(recent_orders.order_count::NUMERIC / NULLIF(all_orders.order_count, 0), 2) AS recent_ratio
FROM customers c
JOIN (
    SELECT 
        customer_id,
        COUNT(*) AS order_count
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'
    AND status = 'completed'
    GROUP BY customer_id
) recent_orders ON recent_orders.customer_id = c.customer_id
JOIN (
    SELECT 
        customer_id,
        COUNT(*) AS order_count
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
) all_orders ON all_orders.customer_id = c.customer_id
WHERE all_orders.order_count >= 2
ORDER BY recent_ratio ASC, total_orders DESC;

-- Query 11: Registration channel performance
-- GROUP BY + aggregation + JOIN
SELECT 
    c.registration_channel,
    COUNT(DISTINCT c.customer_id) AS total_customers,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT o.order_id)::NUMERIC / NULLIF(COUNT(DISTINCT c.customer_id), 0) AS orders_per_customer,
    SUM(o.total_amount) AS total_revenue
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
GROUP BY c.registration_channel
ORDER BY total_revenue DESC NULLS LAST;

-- Query 12: Product cross-sell analysis
-- Complex JOIN + aggregation + GROUP BY
SELECT 
    p1.product_name AS product_a,
    p1.category AS category_a,
    p2.product_name AS product_b,
    p2.category AS category_b,
    COUNT(DISTINCT o.order_id) AS co_occurrence_count
FROM order_items oi1
JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
JOIN products p1 ON p1.product_id = oi1.product_id
JOIN products p2 ON p2.product_id = oi2.product_id
JOIN orders o ON o.order_id = oi1.order_id
WHERE o.status = 'completed'
GROUP BY p1.product_id, p1.product_name, p1.category, p2.product_id, p2.product_name, p2.category
HAVING COUNT(DISTINCT o.order_id) >= 2
ORDER BY co_occurrence_count DESC
LIMIT 20;

