"""Run latency benchmarks for SQL queries.

This script executes each query multiple times (warmup + runs) and measures
execution latency. Results include percentiles (p50, p95) and are saved to JSON.
"""

import os
import sys
import json
import re
import time
import statistics
import argparse
from pathlib import Path
from datetime import datetime
import psycopg2

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SQL_DIR = PROJECT_ROOT / "sql"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics" / "latency"

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


def percentile(data, p):
    """Calculate percentile p (0-100) from sorted data."""
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    return sorted_data[f]


def run_query_benchmark(conn, query_sql, warmup_runs=2, measure_runs=10):
    """Run a query multiple times and measure execution time."""
    latencies = []
    
    with conn.cursor() as cur:
        # Warmup runs (not measured)
        for _ in range(warmup_runs):
            try:
                cur.execute(query_sql)
                cur.fetchall()
            except psycopg2.Error as e:
                print(f"  Error during warmup: {e}")
                return None
        
        # Measurement runs
        for run_num in range(measure_runs):
            try:
                t0 = time.perf_counter()
                cur.execute(query_sql)
                cur.fetchall()
                dt = time.perf_counter() - t0
                latencies.append(dt * 1000)  # Convert to milliseconds
            except psycopg2.Error as e:
                print(f"  Error during run {run_num + 1}: {e}")
                return None
    
    if not latencies:
        return None
    
    # Calculate statistics
    stats = {
        "runs": len(latencies),
        "latencies_ms": latencies,  # Raw latencies for detailed analysis
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }
    
    return stats


def save_results(results_data, output_file):
    """Save benchmark results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run latency benchmarks for SQL queries"
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=SQL_DIR / "queries.sql",
        help="Path to SQL queries file (default: sql/queries.sql)",
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset scale (default: small)",
    )
    parser.add_argument(
        "--index-state",
        choices=["no_index", "with_index"],
        default="no_index",
        help="Index state label (default: no_index)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs per query (default: 2)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of measurement runs per query (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results (default: results/metrics/latency/)",
    )
    args = parser.parse_args()

    # Generate output filename
    output_filename = f"latency_{args.index_state}_{args.scale}.json"
    output_file = args.output_dir / output_filename

    print("=" * 60)
    print("SQL Query Latency Benchmark")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"Queries file: {args.queries_file}")
    print(f"Scale: {args.scale}")
    print(f"Index state: {args.index_state}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Measurement runs: {args.runs}")
    print(f"Output file: {output_file}")
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

    # Prepare results structure
    results = {
        "metadata": {
            "benchmark_timestamp": datetime.now().isoformat(),
            "scale": args.scale,
            "index_state": args.index_state,
            "warmup_runs": args.warmup,
            "measurement_runs": args.runs,
            "database": DB_CONFIG["database"],
            "host": DB_CONFIG["host"],
            "total_queries": len(queries),
        },
        "queries": [],
    }

    # Run benchmarks
    print(f"\nRunning benchmarks...")
    print("-" * 60)
    
    successful = 0
    failed = 0

    for query_info in queries:
        query_num = query_info["number"]
        description = query_info["description"]
        sql_query = query_info["sql"]
        
        print(f"\nQuery {query_num}: {description}")
        print(f"  Running {args.warmup} warmup + {args.runs} measurement runs...")
        
        stats = run_query_benchmark(conn, sql_query, args.warmup, args.runs)
        
        if stats:
            query_result = {
                "query_number": query_num,
                "description": description,
                "sql": sql_query,
                "statistics": stats,
            }
            results["queries"].append(query_result)
            
            print(f"  ✓ Completed: p50={stats['p50_ms']:.2f}ms, p95={stats['p95_ms']:.2f}ms")
            successful += 1
        else:
            print(f"  ✗ Failed")
            failed += 1

    conn.close()

    # Calculate summary statistics
    if results["queries"]:
        all_p50 = [q["statistics"]["p50_ms"] for q in results["queries"]]
        all_p95 = [q["statistics"]["p95_ms"] for q in results["queries"]]
        
        results["summary"] = {
            "total_queries": len(results["queries"]),
            "successful": successful,
            "failed": failed,
            "overall_p50_ms": statistics.median(all_p50),
            "overall_p95_ms": statistics.median(all_p95),
            "min_p50_ms": min(all_p50),
            "max_p50_ms": max(all_p50),
            "min_p95_ms": min(all_p95),
            "max_p95_ms": max(all_p95),
        }

    # Save results
    print("\n" + "=" * 60)
    save_results(results, output_file)
    
    # Print summary
    if results.get("summary"):
        summary = results["summary"]
        print("\nSummary:")
        print(f"  Successful queries: {summary['successful']}/{summary['total_queries']}")
        print(f"  Overall p50: {summary['overall_p50_ms']:.2f}ms")
        print(f"  Overall p95: {summary['overall_p95_ms']:.2f}ms")
        print("=" * 60)


if __name__ == "__main__":
    main()

