"""Generate plots from benchmark results.

This script creates simple graphs to visualize benchmark results:
- Latency vs scale
- Speedup per query
- Throughput vs concurrency
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LATENCY_DIR = PROJECT_ROOT / "results" / "metrics" / "latency"
THROUGHPUT_DIR = PROJECT_ROOT / "results" / "metrics" / "throughput"
SUMMARY_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SCALES = ["small", "medium", "large"]
SCALE_LABELS = {"small": "Small", "medium": "Medium", "large": "Large"}


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


def plot_latency_vs_scale(latency_dir, output_dir):
    """Plot latency vs scale for different index states."""
    scales_order = ["small", "medium", "large"]
    index_states = ["no_index", "with_index"]
    
    # Collect data
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for scale in scales_order:
        for index_state in index_states:
            filename = f"latency_{index_state}_{scale}.json"
            filepath = latency_dir / filename
            
            results = load_json_file(filepath)
            if not results:
                continue
            
            for query in results.get("queries", []):
                query_num = query.get("query_number")
                stats = query.get("statistics", {})
                p50 = stats.get("p50_ms")
                p95 = stats.get("p95_ms")
                
                if query_num and p50 is not None:
                    data[query_num][index_state]["p50"].append((scale, p50))
                if query_num and p95 is not None:
                    data[query_num][index_state]["p95"].append((scale, p95))
    
    if not data:
        print("No latency data found for plotting")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot P50
    ax1 = axes[0]
    for query_num in sorted(data.keys())[:10]:  # Limit to first 10 queries
        query_data = data[query_num]
        for index_state in index_states:
            if index_state not in query_data or "p50" not in query_data[index_state]:
                continue
            
            points = sorted(query_data[index_state]["p50"], key=lambda x: scales_order.index(x[0]))
            scales = [p[0] for p in points]
            latencies = [p[1] for p in points]
            
            label = f"Q{query_num} ({index_state})"
            linestyle = "--" if index_state == "no_index" else "-"
            ax1.plot(scales, latencies, marker="o", label=label, linestyle=linestyle, alpha=0.7)
    
    ax1.set_xlabel("Dataset Scale")
    ax1.set_ylabel("Latency P50 (ms)")
    ax1.set_title("Latency P50 vs Dataset Scale")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot P95
    ax2 = axes[1]
    for query_num in sorted(data.keys())[:10]:  # Limit to first 10 queries
        query_data = data[query_num]
        for index_state in index_states:
            if index_state not in query_data or "p95" not in query_data[index_state]:
                continue
            
            points = sorted(query_data[index_state]["p95"], key=lambda x: scales_order.index(x[0]))
            scales = [p[0] for p in points]
            latencies = [p[1] for p in points]
            
            label = f"Q{query_num} ({index_state})"
            linestyle = "--" if index_state == "no_index" else "-"
            ax2.plot(scales, latencies, marker="s", label=label, linestyle=linestyle, alpha=0.7)
    
    ax2.set_xlabel("Dataset Scale")
    ax2.set_ylabel("Latency P95 (ms)")
    ax2.set_title("Latency P95 vs Dataset Scale")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "latency_vs_scale.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_speedup_per_query(latency_dir, output_dir):
    """Plot speedup per query."""
    scales = ["small", "medium", "large"]
    
    # Collect data
    speedup_data = defaultdict(lambda: defaultdict(dict))
    
    for scale in scales:
        no_index_file = latency_dir / f"latency_no_index_{scale}.json"
        with_index_file = latency_dir / f"latency_with_index_{scale}.json"
        
        no_index_results = load_json_file(no_index_file)
        with_index_results = load_json_file(with_index_file)
        
        if not no_index_results or not with_index_results:
            continue
        
        # Create query lookup
        no_index_queries = {q["query_number"]: q for q in no_index_results.get("queries", [])}
        with_index_queries = {q["query_number"]: q for q in with_index_results.get("queries", [])}
        
        for query_num in set(no_index_queries.keys()) & set(with_index_queries.keys()):
            no_index_p50 = no_index_queries[query_num]["statistics"].get("p50_ms")
            with_index_p50 = with_index_queries[query_num]["statistics"].get("p50_ms")
            
            if no_index_p50 and with_index_p50 and with_index_p50 > 0:
                speedup = no_index_p50 / with_index_p50
                speedup_data[scale][query_num] = speedup
    
    if not speedup_data:
        print("No speedup data found for plotting")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = 0
    x_ticks = []
    x_labels = []
    colors = {"small": "blue", "medium": "green", "large": "red"}
    
    for scale in scales:
        if scale not in speedup_data:
            continue
        
        query_nums = sorted(speedup_data[scale].keys())
        speedups = [speedup_data[scale][q] for q in query_nums]
        
        x_positions = [x_pos + i for i in range(len(query_nums))]
        bars = ax.bar(x_positions, speedups, color=colors[scale], alpha=0.7, label=SCALE_LABELS[scale])
        
        # Add query numbers on bars
        for i, (pos, speedup) in enumerate(zip(x_positions, speedups)):
            ax.text(pos, speedup + 0.05, f"Q{query_nums[i]}", ha="center", va="bottom", fontsize=7)
        
        x_ticks.extend(x_positions)
        x_labels.extend([f"Q{q}" for q in query_nums])
        x_pos += len(query_nums) + 1
    
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="No speedup")
    ax.set_xlabel("Query Number")
    ax.set_ylabel("Speedup (no_index / with_index)")
    ax.set_title("Speedup per Query by Scale")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    output_file = output_dir / "speedup_per_query.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_throughput_vs_concurrency(throughput_dir, output_dir):
    """Plot throughput vs concurrency."""
    # Collect data for different concurrency levels
    # Note: This assumes multiple throughput runs with different concurrency
    # For now, we'll plot QPS vs scale/index_state
    
    scales = ["small", "medium"]
    index_states = ["no_index", "with_index"]
    
    data = defaultdict(lambda: defaultdict(dict))
    
    for scale in scales:
        for index_state in index_states:
            filename = f"throughput_{index_state}_{scale}.json"
            filepath = throughput_dir / filename
            
            results = load_json_file(filepath)
            if not results:
                continue
            
            metadata = results.get("metadata", {})
            concurrency = metadata.get("concurrency", 8)  # Default to 8
            
            # Aggregate QPS across all queries
            qps_values = []
            for query in results.get("queries", []):
                stats = query.get("statistics", {})
                qps = stats.get("qps")
                if qps:
                    qps_values.append(qps)
            
            if qps_values:
                avg_qps = sum(qps_values) / len(qps_values)
                data[scale][index_state] = {
                    "concurrency": concurrency,
                    "avg_qps": avg_qps,
                }
    
    if not data:
        print("No throughput data found for plotting")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = [SCALE_LABELS[s] for s in scales if s in data]
    width = 0.35
    
    no_index_qps = [data[s]["no_index"]["avg_qps"] for s in scales if s in data and "no_index" in data[s]]
    with_index_qps = [data[s]["with_index"]["avg_qps"] for s in scales if s in data and "with_index" in data[s]]
    
    x_pos = list(range(len(x)))
    
    if no_index_qps:
        ax.bar([p - width/2 for p in x_pos], no_index_qps, width, label="No Index", alpha=0.7, color="orange")
    if with_index_qps:
        ax.bar([p + width/2 for p in x_pos], with_index_qps, width, label="With Index", alpha=0.7, color="green")
    
    ax.set_xlabel("Dataset Scale")
    ax.set_ylabel("Average QPS")
    ax.set_title("Throughput (QPS) vs Dataset Scale")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    output_file = output_dir / "throughput_vs_scale.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def main():
    """Main function to generate plots."""
    parser = argparse.ArgumentParser(
        description="Generate plots from benchmark results"
    )
    parser.add_argument(
        "--latency-dir",
        type=Path,
        default=LATENCY_DIR,
        help="Directory containing latency results",
    )
    parser.add_argument(
        "--throughput-dir",
        type=Path,
        default=THROUGHPUT_DIR,
        help="Directory containing throughput results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Output directory for figures",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Benchmark Plots")
    print("=" * 60)
    print(f"Latency directory: {args.latency_dir}")
    print(f"Throughput directory: {args.throughput_dir}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    # Generate plots
    print("\n1. Plotting latency vs scale...")
    plot_latency_vs_scale(args.latency_dir, args.output_dir)
    
    print("\n2. Plotting speedup per query...")
    plot_speedup_per_query(args.latency_dir, args.output_dir)
    
    print("\n3. Plotting throughput vs scale...")
    plot_throughput_vs_concurrency(args.throughput_dir, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Plot generation complete!")
    print(f"Figures saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

