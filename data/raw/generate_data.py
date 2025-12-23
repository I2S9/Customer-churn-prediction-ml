"""Generate synthetic customer churn prediction dataset.

This script generates reproducible synthetic data matching the database schema.
All random operations use a fixed seed for reproducibility.
"""

import random
import csv
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Fixed seed for reproducibility
random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR

# Dataset size parameters by scale
SCALE_PARAMS = {
    "small": {
        "customers": 150,
        "products": 25,
        "orders": 400,
        "interactions": 200,
    },
    "medium": {
        "customers": 1500,
        "products": 250,
        "orders": 4000,
        "interactions": 2000,
    },
    "large": {
        "customers": 15000,
        "products": 2500,
        "orders": 40000,
        "interactions": 20000,
    },
}

# Data generation constants
COUNTRIES = ["USA", "UK", "France", "Germany", "Canada", "Australia", "Spain", "Italy"]
GENDERS = ["M", "F", "Other"]
REGISTRATION_CHANNELS = ["web", "mobile", "email", "referral", "social"]
PRODUCT_CATEGORIES = ["Electronics", "Clothing", "Books", "Home", "Sports", "Toys"]
PRODUCT_NAMES = {
    "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones", "Camera"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Shoes", "Hat"],
    "Books": ["Novel", "Textbook", "Guide", "Biography", "Cookbook"],
    "Home": ["Lamp", "Chair", "Table", "Vase", "Mirror"],
    "Sports": ["Bicycle", "Dumbbells", "Yoga Mat", "Running Shoes", "Tennis Racket"],
    "Toys": ["Puzzle", "Board Game", "Action Figure", "Doll", "Building Blocks"],
}
ORDER_STATUSES = ["completed", "pending", "cancelled", "refunded"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer"]
INTERACTION_TYPES = ["support_ticket", "email", "phone_call", "chat", "complaint"]
INTERACTION_CHANNELS = ["web", "phone", "email", "mobile_app"]
INTERACTION_OUTCOMES = ["resolved", "pending", "escalated", "closed"]


def generate_customers(n_customers):
    """Generate customer records."""
    customers = []
    base_date = datetime(2020, 1, 1)
    
    for customer_id in range(1, n_customers + 1):
        created_at = base_date + timedelta(
            days=random.randint(0, 1095)  # 3 years range
        )
        country = random.choice(COUNTRIES)
        email = f"customer{customer_id}@example.com"
        age = random.randint(18, 80)
        gender = random.choice(GENDERS)
        registration_channel = random.choice(REGISTRATION_CHANNELS)
        
        customers.append({
            "customer_id": customer_id,
            "country": country,
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "email": email,
            "age": age,
            "gender": gender,
            "registration_channel": registration_channel,
        })
    
    return customers


def generate_products(n_products):
    """Generate product records."""
    products = []
    base_date = datetime(2019, 1, 1)
    
    for product_id in range(1, n_products + 1):
        category = random.choice(PRODUCT_CATEGORIES)
        product_name_base = random.choice(PRODUCT_NAMES[category])
        product_name = f"{product_name_base} {product_id}"
        price = round(random.uniform(10.0, 500.0), 2)
        created_at = base_date + timedelta(days=random.randint(0, 365))
        
        products.append({
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
            "price": price,
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    return products


def generate_orders(n_orders, n_customers, products):
    """Generate order records."""
    orders = []
    base_date = datetime(2021, 1, 1)
    product_prices = {p["product_id"]: p["price"] for p in products}
    
    for order_id in range(1, n_orders + 1):
        customer_id = random.randint(1, n_customers)
        order_date = base_date + timedelta(days=random.randint(0, 730))  # 2 years
        status = random.choice(ORDER_STATUSES)
        payment_method = random.choice(PAYMENT_METHODS)
        
        # Generate order items for this order
        n_items = random.randint(1, 5)
        order_items_list = []
        total_amount = 0.0
        
        for _ in range(n_items):
            product_id = random.randint(1, len(products))
            quantity = random.randint(1, 3)
            unit_price = product_prices[product_id]
            total_amount += unit_price * quantity
            
            order_items_list.append({
                "order_item_id": len(orders) * 10 + len(order_items_list) + 1,
                "order_id": order_id,
                "product_id": product_id,
                "quantity": quantity,
                "unit_price": unit_price,
            })
        
        total_amount = round(total_amount, 2)
        
        orders.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "order_date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
            "total_amount": total_amount,
            "status": status,
            "payment_method": payment_method,
        })
        
        # Store order items for later export
        orders[-1]["_items"] = order_items_list
    
    return orders


def generate_interactions(n_interactions, n_customers):
    """Generate customer interaction records."""
    interactions = []
    base_date = datetime(2021, 1, 1)
    
    for interaction_id in range(1, n_interactions + 1):
        customer_id = random.randint(1, n_customers)
        interaction_type = random.choice(INTERACTION_TYPES)
        interaction_date = base_date + timedelta(days=random.randint(0, 730))
        channel = random.choice(INTERACTION_CHANNELS)
        outcome = random.choice(INTERACTION_OUTCOMES)
        
        interactions.append({
            "interaction_id": interaction_id,
            "customer_id": customer_id,
            "interaction_type": interaction_type,
            "interaction_date": interaction_date.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": channel,
            "outcome": outcome,
        })
    
    return interactions


def write_csv(data, filename, fieldnames, scale_suffix=""):
    """Write data to CSV file."""
    if scale_suffix:
        # Add scale suffix before .csv extension
        base_name = filename.replace(".csv", "")
        filename = f"{base_name}_{scale_suffix}.csv"
    
    filepath = DATA_DIR / filename
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Generated {filename}: {len(data)} records")


def main():
    """Generate all synthetic datasets."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic customer churn prediction dataset"
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset scale to generate (default: small)",
    )
    args = parser.parse_args()
    
    # Get scale parameters
    params = SCALE_PARAMS[args.scale]
    n_customers = params["customers"]
    n_products = params["products"]
    n_orders = params["orders"]
    n_interactions = params["interactions"]
    
    print("=" * 60)
    print("Synthetic Customer Churn Dataset Generator")
    print("=" * 60)
    print(f"Scale: {args.scale}")
    print(f"Seed: 42 (fixed for reproducibility)")
    print(f"Customers: {n_customers}")
    print(f"Products: {n_products}")
    print(f"Orders: {n_orders}")
    print(f"Interactions: {n_interactions}")
    print("-" * 60)
    
    # Generate data
    print("\nGenerating data...")
    customers = generate_customers(n_customers)
    products = generate_products(n_products)
    orders = generate_orders(n_orders, n_customers, products)
    interactions = generate_interactions(n_interactions, n_customers)
    
    # Extract order items from orders
    order_items = []
    for order in orders:
        order_items.extend(order.pop("_items", []))
    
    # Write CSV files with scale suffix
    print("\nWriting CSV files...")
    write_csv(
        customers,
        "customers.csv",
        ["customer_id", "country", "created_at", "email", "age", "gender", "registration_channel"],
        scale_suffix=args.scale,
    )
    write_csv(
        products,
        "products.csv",
        ["product_id", "product_name", "category", "price", "created_at"],
        scale_suffix=args.scale,
    )
    write_csv(
        orders,
        "orders.csv",
        ["order_id", "customer_id", "order_date", "total_amount", "status", "payment_method"],
        scale_suffix=args.scale,
    )
    write_csv(
        order_items,
        "order_items.csv",
        ["order_item_id", "order_id", "product_id", "quantity", "unit_price"],
        scale_suffix=args.scale,
    )
    write_csv(
        interactions,
        "customer_interactions.csv",
        ["interaction_id", "customer_id", "interaction_type", "interaction_date", "channel", "outcome"],
        scale_suffix=args.scale,
    )
    
    print("-" * 60)
    print("Dataset generation complete!")
    total_records = len(customers) + len(products) + len(orders) + len(order_items) + len(interactions)
    print(f"Total records: {total_records:,}")
    print(f"Files saved with suffix: _{args.scale}")
    print("=" * 60)


if __name__ == "__main__":
    main()

