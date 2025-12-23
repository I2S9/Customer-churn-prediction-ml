"""Reproduce benchmark experiments from configuration file.

This script reads config.yaml and executes the complete benchmark workflow
to ensure reproducibility across different environments.
"""

import yaml
import subprocess
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def load_config():
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        print(f"Error: Configuration file not found: {CONFIG_FILE}")
        sys.exit(1)
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_env_vars(config):
    """Set environment variables from config."""
    db_config = config.get("database", {})
    os.environ.setdefault("DB_HOST", str(db_config.get("host", "localhost")))
    os.environ.setdefault("DB_PORT", str(db_config.get("port", "5432")))
    os.environ.setdefault("DB_NAME", str(db_config.get("name", "churn_db")))
    os.environ.setdefault("DB_USER", str(db_config.get("user", "postgres")))
    # Password should be set externally for security
    if "DB_PASSWORD" not in os.environ:
        password = db_config.get("password", "")
        if password:
            os.environ["DB_PASSWORD"] = password


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if check and result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        return False
    
    return True


def generate_data(config):
    """Generate synthetic datasets for all scales."""
    scales = config["benchmarks"]["latency"]["scales"]
    
    for scale in scales:
        if not run_command(
            [sys.executable, "data/raw/generate_data.py", "--scale", scale],
            f"Generating {scale} dataset",
        ):
            return False
    return True


def load_data(config):
    """Load datasets into database."""
    scales = config["benchmarks"]["latency"]["scales"]
    
    for scale in scales:
        if not run_command(
            [sys.executable, "benchmarks/load_data.py", "--scale", scale],
            f"Loading {scale} dataset into database",
        ):
            return False
    return True


def run_latency_benchmarks(config):
    """Run latency benchmarks."""
    bench_config = config["benchmarks"]["latency"]
    scales = bench_config["scales"]
    index_states = bench_config["index_states"]
    warmup = bench_config["warmup_runs"]
    runs = bench_config["measurement_runs"]
    
    for scale in scales:
        for index_state in index_states:
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
            if not run_command(
                [
                    sys.executable,
                    "benchmarks/run_benchmarks.py",
                    "--scale", scale,
                    "--index-state", index_state,
                    "--warmup", str(warmup),
                    "--runs", str(runs),
                ],
                f"Running latency benchmark: {scale} scale, {index_state}",
            ):
                continue


def run_throughput_benchmarks(config):
    """Run throughput benchmarks."""
    bench_config = config["benchmarks"]["throughput"]
    scales = bench_config["scales"]
    index_states = bench_config["index_states"]
    concurrency = bench_config["concurrency"]
    duration = bench_config["duration_seconds"]
    
    for scale in scales:
        for index_state in index_states:
            # Apply/drop indexes
            if index_state == "with_index":
                if not run_command(
                    [sys.executable, "benchmarks/apply_indexes.py", "apply"],
                    f"Applying indexes for {scale} scale (throughput)",
                ):
                    continue
            else:
                if not run_command(
                    [sys.executable, "benchmarks/apply_indexes.py", "drop"],
                    f"Dropping indexes for {scale} scale (throughput)",
                ):
                    continue
            
            # Run throughput benchmark
            if not run_command(
                [
                    sys.executable,
                    "benchmarks/run_throughput.py",
                    "--scale", scale,
                    "--index-state", index_state,
                    "--concurrency", str(concurrency),
                    "--duration", str(duration),
                ],
                f"Running throughput benchmark: {scale} scale, {index_state}",
            ):
                continue


def capture_execution_plans(config):
    """Capture execution plans."""
    bench_config = config["benchmarks"]["explain"]
    index_states = bench_config["index_states"]
    
    for index_state in index_states:
        # Apply/drop indexes
        if index_state == "with_index":
            if not run_command(
                [sys.executable, "benchmarks/apply_indexes.py", "apply"],
                f"Applying indexes for execution plans",
            ):
                continue
        else:
            if not run_command(
                [sys.executable, "benchmarks/apply_indexes.py", "drop"],
                f"Dropping indexes for execution plans",
            ):
                continue
        
        # Capture plans
        run_command(
            [
                sys.executable,
                "benchmarks/explain.py",
                "--index-state", index_state,
            ],
            f"Capturing execution plans: {index_state}",
            check=False,  # Don't fail if plans already exist
        )


def analyze_and_plot(config):
    """Analyze results and generate plots."""
    # Analyze results
    run_command(
        [sys.executable, "benchmarks/analyze_results.py"],
        "Analyzing benchmark results",
        check=False,
    )
    
    # Generate plots
    run_command(
        [sys.executable, "benchmarks/plot_results.py"],
        "Generating benchmark plots",
        check=False,
    )


def main():
    """Main function to reproduce benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reproduce benchmark experiments from configuration"
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
        "--skip-latency",
        action="store_true",
        help="Skip latency benchmarks",
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip throughput benchmarks",
    )
    parser.add_argument(
        "--skip-plans",
        action="store_true",
        help="Skip execution plan capture",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis and plotting",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        help="Path to configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    
    global CONFIG_FILE
    CONFIG_FILE = args.config
    
    print("=" * 60)
    print("Benchmark Reproduction Script")
    print("=" * 60)
    print(f"Configuration file: {CONFIG_FILE}")
    print("=" * 60)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    print("Configuration loaded successfully")
    
    # Set environment variables
    set_env_vars(config)
    
    # Execute workflow
    if not args.skip_generate:
        if not generate_data(config):
            print("Data generation failed. Exiting.")
            sys.exit(1)
    
    if not args.skip_load:
        if not load_data(config):
            print("Data loading failed. Exiting.")
            sys.exit(1)
    
    if not args.skip_latency:
        run_latency_benchmarks(config)
    
    if not args.skip_throughput:
        run_throughput_benchmarks(config)
    
    if not args.skip_plans:
        capture_execution_plans(config)
    
    if not args.skip_analysis:
        analyze_and_plot(config)
    
    print("\n" + "=" * 60)
    print("Benchmark reproduction complete!")
    print("=" * 60)
    print("\nResults are available in:")
    print("  - results/metrics/latency/")
    print("  - results/metrics/throughput/")
    print("  - results/metrics/plans/")
    print("  - results/metrics/summary.csv")
    print("  - results/figures/")


if __name__ == "__main__":
    main()

