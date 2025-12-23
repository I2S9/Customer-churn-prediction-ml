# Makefile for Customer Churn Prediction Benchmarks
# Provides convenient shortcuts for common tasks

.PHONY: help install generate load benchmark analyze plot reproduce clean

help:
	@echo "Available targets:"
	@echo "  make install       - Install project dependencies"
	@echo "  make generate      - Generate synthetic datasets for all scales"
	@echo "  make load          - Load datasets into PostgreSQL"
	@echo "  make benchmark     - Run all benchmarks"
	@echo "  make analyze       - Analyze benchmark results"
	@echo "  make plot          - Generate plots from results"
	@echo "  make reproduce     - Run complete reproducible workflow"
	@echo "  make clean         - Clean generated files (use with caution)"

install:
	pip install -e .

generate:
	python data/raw/generate_data.py --scale small
	python data/raw/generate_data.py --scale medium
	python data/raw/generate_data.py --scale large

load:
	python benchmarks/load_data.py --scale small
	python benchmarks/load_data.py --scale medium
	python benchmarks/load_data.py --scale large

benchmark:
	python benchmarks/reproduce_benchmarks.py --skip-generate --skip-load --skip-analysis

analyze:
	python benchmarks/analyze_results.py

plot:
	python benchmarks/plot_results.py

reproduce:
	python benchmarks/reproduce_benchmarks.py

clean:
	@echo "Warning: This will delete generated data and results!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -f data/raw/*.csv; \
		rm -rf results/metrics/*; \
		rm -rf results/figures/*; \
		echo "Cleaned generated files"; \
	else \
		echo "Cancelled"; \
	fi

