-- Customer Churn Prediction Database Schema
-- Version 1.0
--
-- This schema defines the relational structure for customer behavioral data
-- used in churn prediction modeling.

-- Customers table
-- Core customer information and demographics
CREATE TABLE customers (
    customer_id BIGSERIAL PRIMARY KEY,
    country TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    email TEXT,
    age INTEGER,
    gender TEXT,
    registration_channel TEXT
);

-- Products table
-- Product catalog information
CREATE TABLE products (
    product_id BIGSERIAL PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL
);

-- Orders table
-- Customer order transactions
CREATE TABLE orders (
    order_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    order_date TIMESTAMP NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    status TEXT NOT NULL,
    payment_method TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Order items table
-- Individual items within each order
CREATE TABLE order_items (
    order_item_id BIGSERIAL PRIMARY KEY,
    order_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
);

-- Customer interactions table
-- Events and interactions (support tickets, emails, etc.)
CREATE TABLE customer_interactions (
    interaction_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    interaction_type TEXT NOT NULL,
    interaction_date TIMESTAMP NOT NULL,
    channel TEXT,
    outcome TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Note: Indexes are defined separately in sql/indexes.sql
-- This schema represents the baseline state without indexes.
-- Apply indexes using: python benchmarks/apply_indexes.py apply

