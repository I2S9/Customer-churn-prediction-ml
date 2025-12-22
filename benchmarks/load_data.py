"""Load CSV data into PostgreSQL database.

This script loads synthetic datasets into PostgreSQL using COPY for efficient
bulk loading. Database connection is configured via environment variables.
"""

import os
import sys
import io
import argparse
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Database connection parameters from environment
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "churn_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Table definitions with column order matching CSV headers
TABLES = {
    "customers": {
        "file": "customers.csv",
        "columns": [
            "customer_id",
            "country",
            "created_at",
            "email",
            "age",
            "gender",
            "registration_channel",
        ],
    },
    "products": {
        "file": "products.csv",
        "columns": [
            "product_id",
            "product_name",
            "category",
            "price",
            "created_at",
        ],
    },
    "orders": {
        "file": "orders.csv",
        "columns": [
            "order_id",
            "customer_id",
            "order_date",
            "total_amount",
            "status",
            "payment_method",
        ],
    },
    "order_items": {
        "file": "order_items.csv",
        "columns": [
            "order_item_id",
            "order_id",
            "product_id",
            "quantity",
            "unit_price",
        ],
    },
    "customer_interactions": {
        "file": "customer_interactions.csv",
        "columns": [
            "interaction_id",
            "customer_id",
            "interaction_type",
            "interaction_date",
            "channel",
            "outcome",
        ],
    },
}

# Load order respecting foreign key constraints
LOAD_ORDER = [
    "customers",
    "products",
    "orders",
    "order_items",
    "customer_interactions",
]


def get_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


def truncate_table(conn, table_name):
    """Truncate a table, resetting sequences."""
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE")
    conn.commit()
    print(f"  Truncated table: {table_name}")


def load_table_copy(conn, table_name, csv_path, columns):
    """Load data from CSV into table using COPY."""
    with conn.cursor() as cur:
        # Read file and skip header
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Skip first line (header)
            data_lines = "".join(lines[1:])

        # Use copy_expert with StringIO for the data (without header)
        data_stream = io.StringIO(data_lines)
        columns_str = ", ".join(columns)
        copy_sql = f"""
            COPY {table_name} ({columns_str})
            FROM STDIN
            WITH (FORMAT CSV, HEADER false, DELIMITER ',')
        """
        cur.copy_expert(copy_sql, data_stream)
        rowcount = cur.rowcount
    conn.commit()
    return rowcount


def load_table(conn, table_name, csv_path, columns, truncate=True):
    """Load a single table from CSV file."""
    if not csv_path.exists():
        print(f"  Warning: CSV file not found: {csv_path}")
        return 0

    print(f"  Loading {table_name} from {csv_path.name}...")

    if truncate:
        truncate_table(conn, table_name)

    try:
        rowcount = load_table_copy(conn, table_name, csv_path, columns)
        print(f"  Loaded {rowcount} rows into {table_name}")
        return rowcount
    except psycopg2.Error as e:
        print(f"  Error loading {table_name}: {e}")
        conn.rollback()
        raise


def verify_schema(conn):
    """Verify that required tables exist in the database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        existing_tables = {row[0] for row in cur.fetchall()}

    required_tables = set(TABLES.keys())
    missing_tables = required_tables - existing_tables

    if missing_tables:
        print(f"Error: Missing tables in database: {', '.join(missing_tables)}")
        print("Please run sql/schema.sql to create the tables first.")
        return False

    return True


def main():
    """Main function to load data into PostgreSQL."""
    parser = argparse.ArgumentParser(
        description="Load CSV data into PostgreSQL database"
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset scale to load (default: small)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not truncate tables before loading",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PostgreSQL Data Loader")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"Scale: {args.scale}")
    print(f"Data directory: {args.data_dir}")
    print("-" * 60)

    # Connect to database
    print("Connecting to database...")
    conn = get_connection()
    print("Connected successfully")

    # Verify schema
    print("\nVerifying database schema...")
    if not verify_schema(conn):
        conn.close()
        sys.exit(1)
    print("Schema verified")

    # Load tables in correct order
    print(f"\nLoading data (scale: {args.scale})...")
    total_rows = 0

    try:
        for table_name in LOAD_ORDER:
            table_config = TABLES[table_name]
            csv_path = args.data_dir / table_config["file"]

            rowcount = load_table(
                conn,
                table_name,
                csv_path,
                table_config["columns"],
                truncate=not args.no_truncate,
            )
            total_rows += rowcount

        print("-" * 60)
        print(f"Load complete! Total rows loaded: {total_rows}")

    except Exception as e:
        print(f"\nError during load: {e}")
        conn.rollback()
        conn.close()
        sys.exit(1)

    finally:
        conn.close()
        print("Database connection closed")


if __name__ == "__main__":
    main()

