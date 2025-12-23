-- Drop all indexes for customer churn prediction database
-- Version 1.0
--
-- This file drops all indexes to return to baseline (no indexes) state.
-- Useful for benchmarking index impact on query performance.

-- Drop products indexes
DROP INDEX IF EXISTS idx_products_category;

-- Drop customers indexes
DROP INDEX IF EXISTS idx_customers_registration_channel;
DROP INDEX IF EXISTS idx_customers_country;

-- Drop customer interactions indexes
DROP INDEX IF EXISTS idx_customer_interactions_date_type;
DROP INDEX IF EXISTS idx_customer_interactions_channel;
DROP INDEX IF EXISTS idx_customer_interactions_outcome;
DROP INDEX IF EXISTS idx_customer_interactions_type;
DROP INDEX IF EXISTS idx_customer_interactions_date;
DROP INDEX IF EXISTS idx_customer_interactions_customer_id;

-- Drop order items indexes
DROP INDEX IF EXISTS idx_order_items_order_product;
DROP INDEX IF EXISTS idx_order_items_product_id;
DROP INDEX IF EXISTS idx_order_items_order_id;

-- Drop orders indexes
DROP INDEX IF EXISTS idx_orders_status_date;
DROP INDEX IF EXISTS idx_orders_customer_date;
DROP INDEX IF EXISTS idx_orders_status;
DROP INDEX IF EXISTS idx_orders_order_date;
DROP INDEX IF EXISTS idx_orders_customer_id;

