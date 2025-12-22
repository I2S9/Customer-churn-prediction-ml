-- Drop all indexes for customer churn prediction database
-- Version 1.0
--
-- This file drops all indexes to return to baseline (no indexes) state.
-- Useful for benchmarking index impact on query performance.

-- Drop order items indexes
DROP INDEX IF EXISTS idx_order_items_product_id;
DROP INDEX IF EXISTS idx_order_items_order_id;

-- Drop orders indexes
DROP INDEX IF EXISTS idx_orders_order_date;
DROP INDEX IF EXISTS idx_orders_customer_id;

-- Drop customer interactions indexes
DROP INDEX IF EXISTS idx_customer_interactions_date;
DROP INDEX IF EXISTS idx_customer_interactions_customer_id;

