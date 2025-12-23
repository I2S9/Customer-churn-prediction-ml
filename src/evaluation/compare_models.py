"""Compare multiple model evaluation reports.

This script loads evaluation reports and generates a comparison table.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

from src.evaluation.metrics import load_evaluation_report, compare_models

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    """Main function to compare models."""
    parser = argparse.ArgumentParser(
        description="Compare multiple model evaluation reports"
    )
    parser.add_argument(
        "report_files",
        nargs="+",
        type=Path,
        help="Paths to evaluation report JSON files",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Optional model names (default: inferred from filenames)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for comparison CSV (default: reports/model_comparison.csv)",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Which split to compare (default: test)",
    )
    args = parser.parse_args()
    
    if args.output is None:
        args.output = REPORTS_DIR / "model_comparison.csv"
    
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Comparing {len(args.report_files)} models")
    print(f"Split: {args.split}")
    print("-" * 60)
    
    # Load reports
    reports = []
    for report_file in args.report_files:
        if not report_file.exists():
            print(f"Warning: Report file not found: {report_file}")
            continue
        
        report = load_evaluation_report(report_file)
        
        # Extract metrics for the specified split
        eval_data = report.get("evaluation", {})
        
        if args.split in eval_data:
            # Report has separate splits
            split_report = {"metrics": eval_data[args.split]["metrics"]}
        elif "metrics" in eval_data:
            # Single split report (use test by default, or whatever is available)
            split_report = {"metrics": eval_data["metrics"]}
        elif "validation_metrics" in eval_data and args.split == "validation":
            # Validation metrics stored separately
            split_report = {"metrics": eval_data["validation_metrics"]}
        else:
            print(f"Warning: Could not find {args.split} metrics in {report_file}")
            continue
        
        reports.append(split_report)
    
    if not reports:
        print("Error: No valid reports found")
        sys.exit(1)
    
    # Get model names
    if args.model_names:
        model_names = args.model_names[:len(reports)]
    else:
        model_names = [f.stem.replace("_metrics", "") for f in args.report_files[:len(reports)]]
    
    # Compare
    print("\nGenerating comparison...")
    comparison_df = compare_models(reports, model_names)
    
    # Display
    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(args.output, index=False)
    print(f"\nComparison saved to: {args.output}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

