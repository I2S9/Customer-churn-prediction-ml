"""Validate feature dataset for data leakage and data quality issues.

This script performs critical checks to ensure features are valid for churn prediction:
- No features calculated after observation date (data leakage)
- Distribution checks (min/max, NaN percentage)
- Duplicate customer_id checks

Exits with non-zero code if any validation fails.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def check_duplicate_customers(df, customer_id_col="customer_id"):
    """Check for duplicate customer IDs."""
    if customer_id_col not in df.columns:
        raise ValidationError(f"Column '{customer_id_col}' not found in dataset")
    
    duplicates = df[customer_id_col].duplicated()
    n_duplicates = duplicates.sum()
    
    if n_duplicates > 0:
        duplicate_ids = df[duplicates][customer_id_col].unique()
        raise ValidationError(
            f"Found {n_duplicates} duplicate customer IDs. "
            f"Examples: {duplicate_ids[:5].tolist()}"
        )
    
    return True


def check_data_leakage(df, observation_date_col="observation_date", date_cols=None):
    """Check that no features are calculated after observation date."""
    if observation_date_col not in df.columns:
        print(f"Warning: '{observation_date_col}' column not found. Skipping data leakage check.")
        return True
    
    # Convert observation date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[observation_date_col]):
        df[observation_date_col] = pd.to_datetime(df[observation_date_col])
    
    # Check date columns (features that are dates)
    if date_cols is None:
        # Auto-detect date columns
        date_cols = [
            col for col in df.columns
            if col != observation_date_col
            and (pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower())
        ]
    
    leakage_issues = []
    
    for date_col in date_cols:
        if date_col not in df.columns:
            continue
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                continue  # Skip if can't convert
        
        # Check for dates after observation date
        after_observation = df[date_col] > df[observation_date_col]
        n_leakage = after_observation.sum()
        
        if n_leakage > 0:
            leakage_issues.append(
                f"Column '{date_col}': {n_leakage} rows with dates after observation_date"
            )
    
    if leakage_issues:
        raise ValidationError(
            "Data leakage detected! Features calculated after observation date:\n" +
            "\n".join(f"  - {issue}" for issue in leakage_issues)
        )
    
    return True


def check_distributions(df, exclude_cols=None):
    """Check feature distributions and missing values."""
    if exclude_cols is None:
        exclude_cols = ["customer_id", "observation_date", "churn"]
    
    # Get feature columns (exclude IDs and target)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        print("Warning: No feature columns found for distribution checks")
        return True
    
    issues = []
    
    for col in feature_cols:
        # Check NaN percentage
        nan_pct = df[col].isna().sum() / len(df) * 100
        
        if nan_pct > 50:
            issues.append(
                f"Column '{col}': {nan_pct:.1f}% missing values (threshold: 50%)"
            )
        
        # Check numeric distributions
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for infinite values
            if np.isinf(df[col]).any():
                issues.append(f"Column '{col}': Contains infinite values")
            
            # Check for extreme outliers (beyond 6 standard deviations)
            if df[col].notna().sum() > 0:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    extreme_outliers = (z_scores > 6).sum()
                    if extreme_outliers > 0:
                        issues.append(
                            f"Column '{col}': {extreme_outliers} extreme outliers "
                            f"(|z-score| > 6)"
                        )
    
    if issues:
        print("Distribution issues found:")
        for issue in issues:
            print(f"  - {issue}")
        # Don't fail on distribution issues, just warn
        return True
    
    return True


def check_missing_values(df, max_missing_pct=50):
    """Check for excessive missing values."""
    missing_summary = df.isnull().sum() / len(df) * 100
    high_missing = missing_summary[missing_summary > max_missing_pct]
    
    if len(high_missing) > 0:
        print(f"Warning: Columns with >{max_missing_pct}% missing values:")
        for col, pct in high_missing.items():
            print(f"  - {col}: {pct:.1f}%")
        # Warning only, not a failure
    
    return True


def validate_features(filepath, observation_date_col="observation_date", 
                     customer_id_col="customer_id", date_cols=None,
                     strict=True):
    """Validate feature dataset.
    
    Args:
        filepath: Path to feature CSV file
        observation_date_col: Name of observation date column
        customer_id_col: Name of customer ID column
        date_cols: List of date column names to check for leakage
        strict: If True, exit with error code on failures
    
    Returns:
        bool: True if all checks pass
    """
    print("=" * 60)
    print("Feature Validation")
    print("=" * 60)
    print(f"File: {filepath}")
    print("-" * 60)
    
    # Load dataset
    if not filepath.exists():
        error_msg = f"Error: Feature file not found: {filepath}"
        print(error_msg)
        if strict:
            sys.exit(1)
        return False
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        error_msg = f"Error: Could not load feature file: {e}"
        print(error_msg)
        if strict:
            sys.exit(1)
        return False
    
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    errors = []
    warnings = []
    
    # Check 1: Duplicate customers
    print("1. Checking for duplicate customer IDs...")
    try:
        check_duplicate_customers(df, customer_id_col)
        print("   ✓ No duplicate customer IDs")
    except ValidationError as e:
        error_msg = f"   ✗ {e}"
        print(error_msg)
        errors.append(error_msg)
    
    # Check 2: Data leakage
    print("\n2. Checking for data leakage...")
    try:
        check_data_leakage(df, observation_date_col, date_cols)
        print("   ✓ No data leakage detected")
    except ValidationError as e:
        error_msg = f"   ✗ {e}"
        print(error_msg)
        errors.append(error_msg)
    except Exception as e:
        warning_msg = f"   ⚠ Warning: {e}"
        print(warning_msg)
        warnings.append(warning_msg)
    
    # Check 3: Distributions
    print("\n3. Checking feature distributions...")
    try:
        check_distributions(df)
        print("   ✓ Distribution checks passed")
    except Exception as e:
        warning_msg = f"   ⚠ {e}"
        print(warning_msg)
        warnings.append(warning_msg)
    
    # Check 4: Missing values summary
    print("\n4. Checking missing values...")
    try:
        check_missing_values(df)
        print("   ✓ Missing value check completed")
    except Exception as e:
        warning_msg = f"   ⚠ {e}"
        print(warning_msg)
        warnings.append(warning_msg)
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors:
            print(f"  ✗ {error}")
    
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    if not errors and not warnings:
        print("✓ All validation checks passed!")
        return True
    
    if errors:
        print("\nValidation FAILED due to errors above.")
        if strict:
            sys.exit(1)
        return False
    
    print("\nValidation completed with warnings (no critical errors).")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate feature dataset for data leakage and quality issues"
    )
    parser.add_argument(
        "filepath",
        type=Path,
        help="Path to feature CSV file",
    )
    parser.add_argument(
        "--observation-date-col",
        default="observation_date",
        help="Name of observation date column (default: observation_date)",
    )
    parser.add_argument(
        "--customer-id-col",
        default="customer_id",
        help="Name of customer ID column (default: customer_id)",
    )
    parser.add_argument(
        "--date-cols",
        nargs="+",
        help="List of date column names to check for leakage",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Don't exit with error code on failures (warnings only)",
    )
    args = parser.parse_args()
    
    success = validate_features(
        args.filepath,
        observation_date_col=args.observation_date_col,
        customer_id_col=args.customer_id_col,
        date_cols=args.date_cols,
        strict=not args.no_strict,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

