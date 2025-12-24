-- Build base tables for customer churn prediction
-- This script creates the foundational schema for customer and event data

-- Customers table: core customer information
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    signup_date DATE NOT NULL,
    country VARCHAR(50),
    plan_type VARCHAR(20),
    monthly_revenue DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events table: customer behavioral events
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    event_value DECIMAL(10, 2),
    metadata JSON,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_events_customer_id ON events(customer_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_customers_signup_date ON customers(signup_date);

