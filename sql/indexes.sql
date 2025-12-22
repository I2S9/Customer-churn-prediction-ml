-- Indexes for customer churn prediction database
-- Version 1.0
--
-- This file contains all indexes for optimizing query performance.
-- Apply these indexes after loading data for better query performance.

-- Orders table indexes
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);

-- Order items table indexes
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- Customer interactions table indexes
CREATE INDEX idx_customer_interactions_customer_id ON customer_interactions(customer_id);
CREATE INDEX idx_customer_interactions_date ON customer_interactions(interaction_date);

