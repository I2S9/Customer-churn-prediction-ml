"""Run complete benchmark workflow for all scales.

This script automates the process of generating data, loading it, and running
benchmarks for different dataset scales (small, medium, large).
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

SCALES = ["small", "medium", "large"]
INDEX_STATES = ["no_index", "with_index"]


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        return False
    
    return True


def main():
    """Run complete benchmark workflow for all scales."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete benchmark workflow for all dataset scales"
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        choices=SCALES,
        default=SCALES,
        help="Scales to process (default: all)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip data generation step",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip data loading step",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip benchmark execution",
    )
    parser.add_argument(
        "--index-states",
        nargs="+",
        choices=INDEX_STATES,
        default=INDEX_STATES,
        help="Index states to benchmark (default: all)",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Complete Benchmark Workflow")
    print("=" * 60)
    print(f"Scales: {', '.join(args.scales)}")
    print(f"Index states: {', '.join(args.index_states)}")
    print("=" * 60)
    
    for scale in args.scales:
        print(f"\n\n{'#' * 60}")
        print(f"Processing scale: {scale.upper()}")
        print(f"{'#' * 60}")
        
        # Step 1: Generate data
        if not args.skip_generate:
            if not run_command(
                [sys.executable, "data/raw/generate_data.py", "--scale", scale],
                f"Generating {scale} dataset",
            ):
                print(f"Skipping scale {scale} due to generation failure")
                continue
        
        # Step 2: Load data
        if not args.skip_load:
            if not run_command(
                [sys.executable, "benchmarks/load_data.py", "--scale", scale],
                f"Loading {scale} dataset into database",
            ):
                print(f"Skipping scale {scale} due to load failure")
                continue
        
        # Step 3: Run benchmarks for each index state
        if not args.skip_benchmarks:
            for index_state in args.index_states:
                # Apply/drop indexes
                if index_state == "with_index":
                    if not run_command(
                        [sys.executable, "benchmarks/apply_indexes.py", "apply"],
                        f"Applying indexes for {scale} scale",
                    ):
                        continue
                else:
                    if not run_command(
                        [sys.executable, "benchmarks/apply_indexes.py", "drop"],
                        f"Dropping indexes for {scale} scale",
                    ):
                        continue
                
                # Run latency benchmark
                run_command(
                    [
                        sys.executable,
                        "benchmarks/run_benchmarks.py",
                        "--scale", scale,
                        "--index-state", index_state,
                    ],
                    f"Running latency benchmark: {scale} scale, {index_state}",
                )
                
                # Run throughput benchmark (only for small/medium to save time)
                if scale in ["small", "medium"]:
                    run_command(
                        [
                            sys.executable,
                            "benchmarks/run_throughput.py",
                            "--scale", scale,
                            "--index-state", index_state,
                            "--concurrency", "8",
                            "--duration", "30",
                        ],
                        f"Running throughput benchmark: {scale} scale, {index_state}",
                    )
                
                # Capture execution plans
                run_command(
                    [
                        sys.executable,
                        "benchmarks/explain.py",
                        "--index-state", index_state,
                    ],
                    f"Capturing execution plans: {scale} scale, {index_state}",
                )
    
    print("\n" + "=" * 60)
    print("Workflow complete!")
    print("=" * 60)
    print("\nResults are available in:")
    print("  - results/metrics/latency/")
    print("  - results/metrics/throughput/")
    print("  - results/metrics/plans/")


if __name__ == "__main__":
    main()

