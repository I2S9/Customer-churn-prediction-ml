"""Run throughput benchmarks for SQL queries with concurrency.

This script executes queries concurrently to measure throughput (QPS)
under load. Uses ThreadPoolExecutor for concurrent query execution.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from threading import Lock

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SQL_DIR = PROJECT_ROOT / "sql"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics" / "throughput"

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


def execute_query_worker(query_sql, worker_id, start_time, duration_seconds, results_lock, results_list):
    """Worker function that executes queries continuously for the duration."""
    conn = None
    query_count = 0
    error_count = 0
    latencies = []
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            try:
                t0 = time.perf_counter()
                cur.execute(query_sql)
                cur.fetchall()
                dt = time.perf_counter() - t0
                
                latencies.append(dt * 1000)  # Convert to milliseconds
                query_count += 1
            except psycopg2.Error:
                error_count += 1
                # Continue executing despite errors
        
        cur.close()
    except Exception as e:
        print(f"  Worker {worker_id} error: {e}")
    finally:
        if conn:
            conn.close()
    
    # Store results thread-safely
    with results_lock:
        results_list.append({
            "worker_id": worker_id,
            "query_count": query_count,
            "error_count": error_count,
            "latencies_ms": latencies,
        })
    
    return query_count, error_count


def run_throughput_benchmark(query_sql, concurrency, duration_seconds):
    """Run throughput benchmark for a single query with concurrency."""
    start_time = time.time()
    results_lock = Lock()
    results_list = []
    
    # Create thread pool
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all worker tasks
        futures = [
            executor.submit(
                execute_query_worker,
                query_sql,
                worker_id,
                start_time,
                duration_seconds,
                results_lock,
                results_list,
            )
            for worker_id in range(concurrency)
        ]
        
        # Wait for all workers to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  Worker error: {e}")
    
    actual_duration = time.time() - start_time
    
    # Aggregate results
    total_queries = sum(r["query_count"] for r in results_list)
    total_errors = sum(r["error_count"] for r in results_list)
    all_latencies = []
    for r in results_list:
        all_latencies.extend(r["latencies_ms"])
    
    # Calculate statistics
    qps = total_queries / actual_duration if actual_duration > 0 else 0
    
    stats = {
        "concurrency": concurrency,
        "duration_seconds": actual_duration,
        "total_queries": total_queries,
        "total_errors": total_errors,
        "qps": qps,
        "queries_per_worker": total_queries / concurrency if concurrency > 0 else 0,
        "error_rate": total_errors / (total_queries + total_errors) if (total_queries + total_errors) > 0 else 0,
    }
    
    if all_latencies:
        sorted_latencies = sorted(all_latencies)
        n = len(sorted_latencies)
        stats.update({
            "latency_min_ms": min(all_latencies),
            "latency_max_ms": max(all_latencies),
            "latency_mean_ms": statistics.mean(all_latencies),
            "latency_median_ms": statistics.median(all_latencies),
            "latency_p50_ms": sorted_latencies[min(int(n * 0.50), n - 1)],
            "latency_p95_ms": sorted_latencies[min(int(n * 0.95), n - 1)],
            "latency_p99_ms": sorted_latencies[min(int(n * 0.99), n - 1)],
        })
    
    stats["worker_results"] = results_list
    
    return stats


def save_results(results_data, output_file):
    """Save benchmark results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")


def main():
    """Main function to run throughput benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run throughput benchmarks for SQL queries with concurrency"
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
        default="with_index",
        help="Index state label (default: with_index)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        choices=range(1, 17),  # 1-16 workers
        help="Number of concurrent workers (1-16, default: 8)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        choices=range(20, 61),  # 20-60 seconds
        help="Benchmark duration in seconds (20-60, default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results (default: results/metrics/throughput/)",
    )
    args = parser.parse_args()

    # Generate output filename
    output_filename = f"throughput_{args.index_state}_{args.scale}.json"
    output_file = args.output_dir / output_filename

    print("=" * 60)
    print("SQL Query Throughput Benchmark")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"Queries file: {args.queries_file}")
    print(f"Scale: {args.scale}")
    print(f"Index state: {args.index_state}")
    print(f"Concurrency: {args.concurrency} workers")
    print(f"Duration: {args.duration} seconds")
    print(f"Output file: {output_file}")
    print("-" * 60)

    # Parse queries
    print("\nParsing queries...")
    queries = parse_queries(args.queries_file)
    
    if not queries:
        print("No queries found in SQL file.")
        sys.exit(1)
    
    print(f"Found {len(queries)} queries")

    # Prepare results structure
    results = {
        "metadata": {
            "benchmark_timestamp": datetime.now().isoformat(),
            "scale": args.scale,
            "index_state": args.index_state,
            "concurrency": args.concurrency,
            "duration_seconds": args.duration,
            "database": DB_CONFIG["database"],
            "host": DB_CONFIG["host"],
            "total_queries": len(queries),
        },
        "queries": [],
    }

    # Run benchmarks
    print(f"\nRunning throughput benchmarks...")
    print("-" * 60)
    
    successful = 0
    failed = 0

    for query_info in queries:
        query_num = query_info["number"]
        description = query_info["description"]
        sql_query = query_info["sql"]
        
        print(f"\nQuery {query_num}: {description}")
        print(f"  Running with {args.concurrency} workers for {args.duration}s...")
        
        try:
            stats = run_throughput_benchmark(sql_query, args.concurrency, args.duration)
            
            query_result = {
                "query_number": query_num,
                "description": description,
                "sql": sql_query,
                "statistics": stats,
            }
            results["queries"].append(query_result)
            
            print(f"  ✓ Completed: QPS={stats['qps']:.2f}, Total={stats['total_queries']} queries")
            successful += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1

    # Calculate summary statistics
    if results["queries"]:
        all_qps = [q["statistics"]["qps"] for q in results["queries"]]
        all_total_queries = [q["statistics"]["total_queries"] for q in results["queries"]]
        
        results["summary"] = {
            "total_queries": len(results["queries"]),
            "successful": successful,
            "failed": failed,
            "overall_qps": statistics.mean(all_qps),
            "total_queries_executed": sum(all_total_queries),
            "min_qps": min(all_qps),
            "max_qps": max(all_qps),
        }

    # Save results
    print("\n" + "=" * 60)
    save_results(results, output_file)
    
    # Print summary
    if results.get("summary"):
        summary = results["summary"]
        print("\nSummary:")
        print(f"  Successful queries: {summary['successful']}/{summary['total_queries']}")
        print(f"  Overall QPS: {summary['overall_qps']:.2f}")
        print(f"  Total queries executed: {summary['total_queries_executed']}")
        print("=" * 60)


if __name__ == "__main__":
    main()

