"""Capture execution plans for SQL queries using EXPLAIN.

This script executes EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) for each query
in sql/queries.sql and saves the execution plans as JSON files.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
import psycopg2

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SQL_DIR = PROJECT_ROOT / "sql"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics" / "plans"

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


def parse_queries(sql_file):
    """Parse SQL file and extract individual queries."""
    if not sql_file.exists():
        print(f"Error: SQL file not found: {sql_file}")
        return []

    with open(sql_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by query markers (-- Query N:)
    query_pattern = r"-- Query (\d+):\s*(.+?)(?=\n-- Query \d+:|$)"
    matches = re.finditer(query_pattern, content, re.DOTALL)

    queries = []
    for match in matches:
        query_num = int(match.group(1))
        query_section = match.group(2)
        
        # Extract query description (first line after Query N:)
        description_match = re.match(r"^(.+?)\n", query_section)
        description = description_match.group(1).strip() if description_match else ""
        
        # Extract SQL query (everything after description and comments)
        # Remove comment lines and get the actual SQL
        sql_lines = []
        for line in query_section.split("\n"):
            line = line.strip()
            # Skip empty lines and comment-only lines
            if line and not line.startswith("--"):
                sql_lines.append(line)
        
        sql_query = " ".join(sql_lines)
        
        if sql_query:
            queries.append({
                "number": query_num,
                "description": description,
                "sql": sql_query,
            })

    return queries


def explain_query(conn, query_sql):
    """Execute EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) and return result."""
    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query_sql}"
    
    try:
        with conn.cursor() as cur:
            cur.execute(explain_sql)
            result = cur.fetchone()
            if result and result[0]:
                return result[0]
            return None
    except psycopg2.Error as e:
        print(f"  Error executing EXPLAIN: {e}")
        return None


def save_plan(plan_data, output_file, query_info):
    """Save execution plan to JSON file with metadata."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plan_document = {
        "metadata": {
            "query_number": query_info["number"],
            "query_description": query_info["description"],
            "captured_at": datetime.now().isoformat(),
            "explain_options": ["ANALYZE", "BUFFERS", "FORMAT JSON"],
        },
        "plan": plan_data,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(plan_document, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved: {output_file.name}")


def main():
    """Main function to capture execution plans."""
    parser = argparse.ArgumentParser(
        description="Capture execution plans for SQL queries"
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=SQL_DIR / "queries.sql",
        help="Path to SQL queries file (default: sql/queries.sql)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "no_index",
        help="Output directory for plan JSON files (default: results/metrics/plans/no_index/)",
    )
    parser.add_argument(
        "--index-state",
        choices=["no_index", "with_index"],
        default="no_index",
        help="Index state label for output directory (default: no_index)",
    )
    args = parser.parse_args()

    # Override output dir if index_state is specified
    if args.index_state:
        args.output_dir = RESULTS_DIR / args.index_state

    print("=" * 60)
    print("SQL Execution Plan Capture")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"Queries file: {args.queries_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Index state: {args.index_state}")
    print("-" * 60)

    # Parse queries
    print("\nParsing queries...")
    queries = parse_queries(args.queries_file)
    
    if not queries:
        print("No queries found in SQL file.")
        sys.exit(1)
    
    print(f"Found {len(queries)} queries")

    # Connect to database
    print("\nConnecting to database...")
    conn = get_connection()
    print("Connected successfully")

    # Capture plans for each query
    print(f"\nCapturing execution plans...")
    print("-" * 60)
    
    successful = 0
    failed = 0

    for query_info in queries:
        query_num = query_info["number"]
        description = query_info["description"]
        sql_query = query_info["sql"]
        
        print(f"\nQuery {query_num}: {description}")
        
        plan_data = explain_query(conn, sql_query)
        
        if plan_data:
            output_file = args.output_dir / f"query_{query_num:02d}.json"
            save_plan(plan_data, output_file, query_info)
            successful += 1
        else:
            print(f"  Failed to capture plan for Query {query_num}")
            failed += 1

    conn.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total queries: {len(queries)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Plans saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

