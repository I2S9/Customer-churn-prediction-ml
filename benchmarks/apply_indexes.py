"""Apply or drop database indexes.

This script allows toggling indexes on/off for benchmarking purposes.
It can apply indexes from sql/indexes.sql or drop them using sql/drop_indexes.sql.
"""

import os
import sys
import argparse
from pathlib import Path
import psycopg2

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SQL_DIR = PROJECT_ROOT / "sql"

# Database connection parameters from environment
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "churn_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def get_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


def execute_sql_file(conn, sql_file):
    """Execute SQL commands from a file."""
    if not sql_file.exists():
        print(f"Error: SQL file not found: {sql_file}")
        return False

    print(f"Executing SQL file: {sql_file.name}")

    with conn.cursor() as cur:
        with open(sql_file, "r", encoding="utf-8") as f:
            sql_content = f.read()
            # Execute all statements
            cur.execute(sql_content)

    conn.commit()
    return True


def list_indexes(conn):
    """List all indexes in the database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                schemaname,
                tablename,
                indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND indexname NOT LIKE 'pg_%'
            ORDER BY tablename, indexname
        """)
        indexes = cur.fetchall()
    return indexes


def main():
    """Main function to apply or drop indexes."""
    parser = argparse.ArgumentParser(
        description="Apply or drop database indexes"
    )
    parser.add_argument(
        "action",
        choices=["apply", "drop", "status"],
        help="Action to perform: apply indexes, drop indexes, or show status",
    )
    parser.add_argument(
        "--sql-dir",
        type=Path,
        default=SQL_DIR,
        help="Directory containing SQL files (default: sql/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Database Index Manager")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"User: {DB_CONFIG['user']}")
    print("-" * 60)

    # Connect to database
    print("Connecting to database...")
    conn = get_connection()
    print("Connected successfully\n")

    if args.action == "status":
        print("Current indexes in database:")
        print("-" * 60)
        indexes = list_indexes(conn)
        if indexes:
            for schema, table, index in indexes:
                print(f"  {table}.{index}")
            print(f"\nTotal: {len(indexes)} indexes")
        else:
            print("  No indexes found (baseline state)")
        conn.close()
        return

    # Apply or drop indexes
    if args.action == "apply":
        sql_file = args.sql_dir / "indexes.sql"
        if execute_sql_file(conn, sql_file):
            print("Indexes applied successfully")
        else:
            conn.close()
            sys.exit(1)

    elif args.action == "drop":
        sql_file = args.sql_dir / "drop_indexes.sql"
        if execute_sql_file(conn, sql_file):
            print("Indexes dropped successfully")
        else:
            conn.close()
            sys.exit(1)

    # Show final status
    print("\nFinal index status:")
    print("-" * 60)
    indexes = list_indexes(conn)
    if indexes:
        for schema, table, index in indexes:
            print(f"  {table}.{index}")
        print(f"\nTotal: {len(indexes)} indexes")
    else:
        print("  No indexes (baseline state)")

    conn.close()
    print("\nDatabase connection closed")


if __name__ == "__main__":
    main()

