# Reproducibility Guide

This document describes how to reproduce the benchmark experiments in this project.

## Prerequisites

1. **Python**: Version 3.8 or higher
2. **PostgreSQL**: Version 12 or higher
3. **Dependencies**: Install project dependencies

```bash
pip install -e .
```

## Configuration

All benchmark parameters are defined in `config.yaml`. This file contains:

- Data generation parameters (seed, scale sizes)
- Database connection settings
- Benchmark parameters (warmup runs, measurement runs, etc.)
- Paths to data and results directories

### Environment Variables

Database password should be set via environment variable for security:

```bash
export DB_PASSWORD=your_password
```

Other database settings can be overridden via environment variables:
- `DB_HOST` (default: localhost)
- `DB_PORT` (default: 5432)
- `DB_NAME` (default: churn_db)
- `DB_USER` (default: postgres)

## Reproducing Experiments

### Quick Start

Run the complete benchmark workflow:

```bash
python benchmarks/reproduce_benchmarks.py
```

This will:
1. Generate synthetic datasets for all scales
2. Load data into PostgreSQL
3. Run latency benchmarks (no_index and with_index)
4. Run throughput benchmarks
5. Capture execution plans
6. Analyze results and generate plots

### Step-by-Step

You can also run individual steps:

```bash
# 1. Generate data
python data/raw/generate_data.py --scale small
python data/raw/generate_data.py --scale medium
python data/raw/generate_data.py --scale large

# 2. Load data
python benchmarks/load_data.py --scale small
python benchmarks/load_data.py --scale medium
python benchmarks/load_data.py --scale large

# 3. Run benchmarks
python benchmarks/run_benchmarks.py --scale small --index-state no_index
python benchmarks/run_benchmarks.py --scale small --index-state with_index
# ... (repeat for other scales)

# 4. Analyze and plot
python benchmarks/analyze_results.py
python benchmarks/plot_results.py
```

### Skipping Steps

You can skip steps if data already exists:

```bash
python benchmarks/reproduce_benchmarks.py \
    --skip-generate \
    --skip-load \
    --skip-throughput
```

## Reproducibility Guarantees

### Fixed Seeds

- **Data generation**: Seed is fixed to `42` in `data/raw/generate_data.py`
- All random operations use this seed for deterministic data generation

### Version Control

- All code is version-controlled in Git
- Configuration is stored in `config.yaml`
- Results include timestamps and metadata

### Deterministic Operations

- SQL queries are deterministic
- Database operations use explicit transactions
- File paths are relative and consistent

## Expected Results

When reproducing experiments, you should get:

- **Identical data**: Same synthetic datasets (same seed)
- **Similar performance**: Latency and throughput should be within normal variance
- **Same structure**: Results files have the same format and structure

Note: Exact latency values may vary slightly due to:
- System load
- Database cache state
- Hardware differences

However, the relative performance (speedup ratios) should be consistent.

## Troubleshooting

### Database Connection Issues

Ensure PostgreSQL is running and accessible:

```bash
psql -h localhost -U postgres -d churn_db
```

### Missing Dependencies

Install all dependencies:

```bash
pip install -e ".[dev]"
```

### Configuration Errors

Check `config.yaml` syntax:

```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

## File Structure

```
results/
  metrics/
    latency/          # Latency benchmark results
    throughput/       # Throughput benchmark results
    plans/            # Execution plans
    summary.csv       # Aggregated results
  figures/            # Generated plots
```

## Version Information

- Python: >=3.8
- PostgreSQL: >=12.0
- psycopg2-binary: >=2.9.0
- matplotlib: >=3.5.0
- pyyaml: >=6.0

For exact versions, check `pyproject.toml`.

