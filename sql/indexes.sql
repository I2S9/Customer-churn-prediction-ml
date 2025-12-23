-- Indexes for customer churn prediction database
-- Version 1.0
--
-- This file contains all indexes for optimizing query performance.
-- Apply these indexes after loading data for better query performance.

-- Orders table indexes
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
-- Filter column: frequently used in WHERE clauses (status = 'completed')
CREATE INDEX idx_orders_status ON orders(status);
-- Composite index: optimizes queries filtering by customer and date
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
-- Composite index: optimizes queries filtering by status and date (monthly trends)
CREATE INDEX idx_orders_status_date ON orders(status, order_date);

-- Order items table indexes
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
-- Composite index: optimizes cross-sell analysis queries
CREATE INDEX idx_order_items_order_product ON order_items(order_id, product_id);

-- Customer interactions table indexes
CREATE INDEX idx_customer_interactions_customer_id ON customer_interactions(customer_id);
CREATE INDEX idx_customer_interactions_date ON customer_interactions(interaction_date);
-- Filter/group columns: frequently used in GROUP BY and WHERE clauses
CREATE INDEX idx_customer_interactions_type ON customer_interactions(interaction_type);
CREATE INDEX idx_customer_interactions_outcome ON customer_interactions(outcome);
CREATE INDEX idx_customer_interactions_channel ON customer_interactions(channel);
-- Composite index: optimizes queries filtering by date and grouping by type/outcome
CREATE INDEX idx_customer_interactions_date_type ON customer_interactions(interaction_date, interaction_type);

-- Customers table indexes
-- Group column: frequently used in GROUP BY clauses
CREATE INDEX idx_customers_country ON customers(country);
CREATE INDEX idx_customers_registration_channel ON customers(registration_channel);

-- Products table indexes
-- Group column: frequently used in GROUP BY clauses
CREATE INDEX idx_products_category ON products(category);

