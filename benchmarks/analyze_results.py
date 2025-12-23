"""Analyze and aggregate benchmark results.

This script reads latency and throughput benchmark results and generates
a comparative summary CSV with p50/p95 metrics and speedup calculations.
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LATENCY_DIR = PROJECT_ROOT / "results" / "metrics" / "latency"
THROUGHPUT_DIR = PROJECT_ROOT / "results" / "metrics" / "throughput"
SUMMARY_DIR = PROJECT_ROOT / "results" / "metrics"

SCALES = ["small", "medium", "large"]
INDEX_STATES = ["no_index", "with_index"]


def load_json_file(filepath):
    """Load JSON file and return data."""
    if not filepath.exists():
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def calculate_speedup(no_index_value, with_index_value):
    """Calculate speedup: no_index / with_index."""
    if with_index_value is None or with_index_value == 0:
        return None
    if no_index_value is None:
        return None
    return no_index_value / with_index_value


def extract_latency_metrics(results_data, query_number):
    """Extract latency metrics for a specific query."""
    if not results_data or "queries" not in results_data:
        return None
    
    for query in results_data["queries"]:
        if query.get("query_number") == query_number:
            stats = query.get("statistics", {})
            return {
                "p50_ms": stats.get("p50_ms"),
                "p95_ms": stats.get("p95_ms"),
                "mean_ms": stats.get("mean_ms"),
            }
    return None


def extract_throughput_metrics(results_data, query_number):
    """Extract throughput metrics for a specific query."""
    if not results_data or "queries" not in results_data:
        return None
    
    for query in results_data["queries"]:
        if query.get("query_number") == query_number:
            stats = query.get("statistics", {})
            return {
                "qps": stats.get("qps"),
                "latency_p50_ms": stats.get("latency_p50_ms"),
                "latency_p95_ms": stats.get("latency_p95_ms"),
            }
    return None


def get_query_description(results_data, query_number):
    """Get query description from results data."""
    if not results_data or "queries" not in results_data:
        return f"Query {query_number}"
    
    for query in results_data["queries"]:
        if query.get("query_number") == query_number:
            return query.get("description", f"Query {query_number}")
    return f"Query {query_number}"


def analyze_latency_results():
    """Analyze latency benchmark results."""
    latency_data = defaultdict(lambda: defaultdict(dict))
    
    for scale in SCALES:
        for index_state in INDEX_STATES:
            filename = f"latency_{index_state}_{scale}.json"
            filepath = LATENCY_DIR / filename
            
            results = load_json_file(filepath)
            if not results:
                continue
            
            # Extract metrics for each query
            for query in results.get("queries", []):
                query_num = query.get("query_number")
                if query_num is None:
                    continue
                
                stats = query.get("statistics", {})
                latency_data[scale][query_num][index_state] = {
                    "p50_ms": stats.get("p50_ms"),
                    "p95_ms": stats.get("p95_ms"),
                    "mean_ms": stats.get("mean_ms"),
                    "description": query.get("description", f"Query {query_num}"),
                }
    
    return latency_data


def analyze_throughput_results():
    """Analyze throughput benchmark results."""
    throughput_data = defaultdict(lambda: defaultdict(dict))
    
    for scale in SCALES:
        for index_state in INDEX_STATES:
            filename = f"throughput_{index_state}_{scale}.json"
            filepath = THROUGHPUT_DIR / filename
            
            results = load_json_file(filepath)
            if not results:
                continue
            
            # Extract metrics for each query
            for query in results.get("queries", []):
                query_num = query.get("query_number")
                if query_num is None:
                    continue
                
                stats = query.get("statistics", {})
                throughput_data[scale][query_num][index_state] = {
                    "qps": stats.get("qps"),
                    "latency_p50_ms": stats.get("latency_p50_ms"),
                    "latency_p95_ms": stats.get("latency_p95_ms"),
                    "description": query.get("description", f"Query {query_num}"),
                }
    
    return throughput_data


def generate_summary_csv(latency_data, throughput_data, output_file):
    """Generate summary CSV with aggregated metrics."""
    rows = []
    
    # Process latency data
    for scale in sorted(SCALES):
        if scale not in latency_data:
            continue
        
        for query_num in sorted(latency_data[scale].keys()):
            query_data = latency_data[scale][query_num]
            no_index = query_data.get("no_index", {})
            with_index = query_data.get("with_index", {})
            
            description = no_index.get("description") or with_index.get("description") or f"Query {query_num}"
            
            p50_no_index = no_index.get("p50_ms")
            p95_no_index = no_index.get("p95_ms")
            p50_with_index = with_index.get("p50_ms")
            p95_with_index = with_index.get("p95_ms")
            
            speedup_p50 = calculate_speedup(p50_no_index, p50_with_index)
            speedup_p95 = calculate_speedup(p95_no_index, p95_with_index)
            
            rows.append({
                "query_number": query_num,
                "query_description": description,
                "scale": scale,
                "metric_type": "latency",
                "p50_no_index_ms": round(p50_no_index, 2) if p50_no_index is not None else None,
                "p95_no_index_ms": round(p95_no_index, 2) if p95_no_index is not None else None,
                "p50_with_index_ms": round(p50_with_index, 2) if p50_with_index is not None else None,
                "p95_with_index_ms": round(p95_with_index, 2) if p95_with_index is not None else None,
                "speedup_p50": round(speedup_p50, 2) if speedup_p50 is not None else None,
                "speedup_p95": round(speedup_p95, 2) if speedup_p95 is not None else None,
                "qps_no_index": None,
                "qps_with_index": None,
                "qps_speedup": None,
            })
    
    # Process throughput data
    for scale in sorted(SCALES):
        if scale not in throughput_data:
            continue
        
        for query_num in sorted(throughput_data[scale].keys()):
            query_data = throughput_data[scale][query_num]
            no_index = query_data.get("no_index", {})
            with_index = query_data.get("with_index", {})
            
            description = no_index.get("description") or with_index.get("description") or f"Query {query_num}"
            
            qps_no_index = no_index.get("qps")
            qps_with_index = with_index.get("qps")
            qps_speedup = calculate_speedup(qps_no_index, qps_with_index)
            
            latency_p50_no_index = no_index.get("latency_p50_ms")
            latency_p95_no_index = no_index.get("latency_p95_ms")
            latency_p50_with_index = with_index.get("latency_p50_ms")
            latency_p95_with_index = with_index.get("latency_p95_ms")
            
            rows.append({
                "query_number": query_num,
                "query_description": description,
                "scale": scale,
                "metric_type": "throughput",
                "p50_no_index_ms": round(latency_p50_no_index, 2) if latency_p50_no_index is not None else None,
                "p95_no_index_ms": round(latency_p95_no_index, 2) if latency_p95_no_index is not None else None,
                "p50_with_index_ms": round(latency_p50_with_index, 2) if latency_p50_with_index is not None else None,
                "p95_with_index_ms": round(latency_p95_with_index, 2) if latency_p95_with_index is not None else None,
                "speedup_p50": round(calculate_speedup(latency_p50_no_index, latency_p50_with_index), 2) if calculate_speedup(latency_p50_no_index, latency_p50_with_index) is not None else None,
                "speedup_p95": round(calculate_speedup(latency_p95_no_index, latency_p95_with_index), 2) if calculate_speedup(latency_p95_no_index, latency_p95_with_index) is not None else None,
                "qps_no_index": round(qps_no_index, 2) if qps_no_index is not None else None,
                "qps_with_index": round(qps_with_index, 2) if qps_with_index is not None else None,
                "qps_speedup": round(qps_speedup, 2) if qps_speedup is not None else None,
            })
    
    # Write CSV
    if not rows:
        print("Warning: No data to write to CSV")
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "query_number",
        "query_description",
        "scale",
        "metric_type",
        "p50_no_index_ms",
        "p95_no_index_ms",
        "p50_with_index_ms",
        "p95_with_index_ms",
        "speedup_p50",
        "speedup_p95",
        "qps_no_index",
        "qps_with_index",
        "qps_speedup",
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Summary CSV written: {output_file}")
    print(f"Total rows: {len(rows)}")


def main():
    """Main function to analyze results."""
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate summary CSV"
    )
    parser.add_argument(
        "--latency-dir",
        type=Path,
        default=LATENCY_DIR,
        help="Directory containing latency results (default: results/metrics/latency/)",
    )
    parser.add_argument(
        "--throughput-dir",
        type=Path,
        default=THROUGHPUT_DIR,
        help="Directory containing throughput results (default: results/metrics/throughput/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SUMMARY_DIR / "summary.csv",
        help="Output CSV file (default: results/metrics/summary.csv)",
    )
    args = parser.parse_args()
    
    global LATENCY_DIR, THROUGHPUT_DIR
    LATENCY_DIR = args.latency_dir
    THROUGHPUT_DIR = args.throughput_dir
    
    print("=" * 60)
    print("Benchmark Results Analysis")
    print("=" * 60)
    print(f"Latency directory: {LATENCY_DIR}")
    print(f"Throughput directory: {THROUGHPUT_DIR}")
    print(f"Output file: {args.output}")
    print("-" * 60)
    
    # Analyze results
    print("\nAnalyzing latency results...")
    latency_data = analyze_latency_results()
    print(f"Found latency data for {len(latency_data)} scales")
    
    print("\nAnalyzing throughput results...")
    throughput_data = analyze_throughput_results()
    print(f"Found throughput data for {len(throughput_data)} scales")
    
    # Generate summary CSV
    print("\nGenerating summary CSV...")
    generate_summary_csv(latency_data, throughput_data, args.output)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

