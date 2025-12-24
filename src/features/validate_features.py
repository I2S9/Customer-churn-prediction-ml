"""Validate features for data leakage and data quality issues."""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures."""

    pass


def check_duplicates(df: pd.DataFrame) -> None:
    """
    Check for duplicate customer_id values.

    Parameters
    ----------
    df : pandas.DataFrame
        Features DataFrame.

    Raises
    ------
    ValidationError
        If duplicate customer_ids are found.
    """
    logger.info("Checking for duplicate customer_ids...")
    duplicates = df["customer_id"].duplicated().sum()

    if duplicates > 0:
        duplicate_ids = df[df["customer_id"].duplicated()]["customer_id"].unique()
        error_msg = (
            f"Found {duplicates} duplicate customer_ids: "
            f"{', '.join(map(str, duplicate_ids[:10]))}"
            + (f" and {len(duplicate_ids) - 10} more" if len(duplicate_ids) > 10 else "")
        )
        logger.error(error_msg)
        raise ValidationError(error_msg)

    logger.info("✓ No duplicate customer_ids found")


def check_missing_values(df: pd.DataFrame, max_nan_pct: float = 50.0) -> None:
    """
    Check for excessive missing values in features.

    Parameters
    ----------
    df : pandas.DataFrame
        Features DataFrame.
    max_nan_pct : float, default 50.0
        Maximum allowed percentage of NaN values per column.

    Raises
    ------
    ValidationError
        If any column has more than max_nan_pct missing values.
    """
    logger.info(f"Checking for missing values (max {max_nan_pct}% per column)...")

    nan_pct = (df.isnull().sum() / len(df)) * 100
    high_nan_cols = nan_pct[nan_pct > max_nan_pct]

    if len(high_nan_cols) > 0:
        error_msg = f"Columns with >{max_nan_pct}% missing values:\n"
        for col, pct in high_nan_cols.items():
            error_msg += f"  {col}: {pct:.2f}%\n"
        logger.error(error_msg)
        raise ValidationError(error_msg.strip())

    total_nan = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    total_nan_pct = (total_nan / total_cells) * 100

    logger.info(f"✓ Missing values check passed (total: {total_nan_pct:.2f}%)")


def check_distributions(df: pd.DataFrame) -> None:
    """
    Check feature distributions for suspicious values (infinite, extreme).

    Parameters
    ----------
    df : pandas.DataFrame
        Features DataFrame.

    Raises
    ------
    ValidationError
        If infinite values or extreme outliers are found.
    """
    logger.info("Checking feature distributions...")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["customer_id", "label"]]

    issues = []

    # Check for infinite values
    for col in numeric_cols:
        inf_count = (df[col] == float("inf")).sum() + (df[col] == float("-inf")).sum()
        if inf_count > 0:
            issues.append(f"  {col}: {inf_count} infinite values")

    if issues:
        error_msg = "Found infinite values in features:\n" + "\n".join(issues)
        logger.error(error_msg)
        raise ValidationError(error_msg)

    # Check for extreme outliers (values > 1e10 or < -1e10)
    for col in numeric_cols:
        extreme_high = (df[col] > 1e10).sum()
        extreme_low = (df[col] < -1e10).sum()
        if extreme_high > 0 or extreme_low > 0:
            issues.append(
                f"  {col}: {extreme_high} values > 1e10, {extreme_low} values < -1e10"
            )

    if issues:
        error_msg = "Found extreme outliers in features:\n" + "\n".join(issues)
        logger.error(error_msg)
        raise ValidationError(error_msg)

    # Log summary statistics
    logger.info("Feature distribution summary:")
    for col in numeric_cols[:10]:  # Show first 10 columns
        logger.info(
            f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
            f"mean={df[col].mean():.2f}, std={df[col].std():.2f}"
        )
    if len(numeric_cols) > 10:
        logger.info(f"  ... and {len(numeric_cols) - 10} more columns")

    logger.info("✓ Distribution check passed")


def check_label_distribution(df: pd.DataFrame, min_samples_per_class: int = 10) -> None:
    """
    Check label distribution for class imbalance issues.

    Parameters
    ----------
    df : pandas.DataFrame
        Features DataFrame with 'label' column.
    min_samples_per_class : int, default 10
        Minimum number of samples required per class.

    Raises
    ------
    ValidationError
        If label distribution is invalid.
    """
    logger.info("Checking label distribution...")

    if "label" not in df.columns:
        error_msg = "Label column 'label' not found in DataFrame"
        logger.error(error_msg)
        raise ValidationError(error_msg)

    label_counts = df["label"].value_counts()
    logger.info(f"Label distribution:\n{label_counts}")

    # Check for valid label values (0 or 1)
    invalid_labels = df[~df["label"].isin([0, 1])]
    if len(invalid_labels) > 0:
        error_msg = f"Found {len(invalid_labels)} invalid label values (not 0 or 1)"
        logger.error(error_msg)
        raise ValidationError(error_msg)

    # Check minimum samples per class
    if len(label_counts) < 2:
        error_msg = f"Only {len(label_counts)} class(es) found, expected 2"
        logger.error(error_msg)
        raise ValidationError(error_msg)

    for label, count in label_counts.items():
        if count < min_samples_per_class:
            error_msg = (
                f"Class {label} has only {count} samples, "
                f"minimum required: {min_samples_per_class}"
            )
            logger.error(error_msg)
            raise ValidationError(error_msg)

    churn_rate = (df["label"] == 1).sum() / len(df) * 100
    logger.info(f"Churn rate: {churn_rate:.2f}%")
    logger.info("✓ Label distribution check passed")


def check_data_leakage(
    df: pd.DataFrame,
    observation_date_col: Optional[str] = None,
) -> None:
    """
    Check for potential data leakage.

    This is a placeholder for data leakage checks. In a real scenario,
    this would verify that no features use information from after the
    observation date.

    Parameters
    ----------
    df : pandas.DataFrame
        Features DataFrame.
    observation_date_col : str, optional
        Column name containing observation date (if available).

    Notes
    -----
    Data leakage checks are conceptual here. In practice, you would:
    1. Verify that all event-based features only use events before observation_date
    2. Check that no future information is included in features
    3. Validate temporal ordering of features

    For now, we log a warning that manual review is recommended.
    """
    logger.info("Checking for data leakage...")

    # Check for suspiciously high predictive power (potential leakage indicator)
    if "label" in df.columns:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        numeric_cols = [c for c in numeric_cols if c not in ["customer_id", "label"]]

        # Check for perfect or near-perfect correlations with label
        high_corr_cols = []
        for col in numeric_cols[:20]:  # Check first 20 columns
            try:
                corr = abs(df[col].corr(df["label"]))
                if corr > 0.95:
                    high_corr_cols.append(f"  {col}: correlation={corr:.3f}")
            except Exception:
                pass

        if high_corr_cols:
            logger.warning(
                "Found features with very high correlation with label "
                "(potential data leakage):\n" + "\n".join(high_corr_cols)
            )
            logger.warning(
                "Manual review recommended to ensure no future information "
                "is included in features"
            )

    logger.info(
        "✓ Data leakage check passed (manual review of feature computation "
        "recommended for production)"
    )


def validate_features(
    features_path: str | Path,
    strict: bool = True,
) -> bool:
    """
    Validate features DataFrame for quality and data leakage.

    Parameters
    ----------
    features_path : str | Path
        Path to features Parquet file.
    strict : bool, default True
        If True, raise ValidationError on any issue. If False, log warnings.

    Returns
    -------
    bool
        True if validation passed, False otherwise.

    Raises
    ------
    ValidationError
        If strict=True and validation fails.
    """
    features_path = Path(features_path)

    if not features_path.exists():
        error_msg = f"Features file not found: {features_path}"
        logger.error(error_msg)
        if strict:
            raise ValidationError(error_msg)
        return False

    logger.info("=" * 60)
    logger.info("Feature Validation")
    logger.info("=" * 60)
    logger.info(f"Loading features from: {features_path}")

    try:
        df = pd.read_parquet(features_path)
    except Exception as e:
        error_msg = f"Failed to load features file: {e}"
        logger.error(error_msg)
        if strict:
            raise ValidationError(error_msg) from e
        return False

    logger.info(f"Features shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Required columns
    if "customer_id" not in df.columns:
        error_msg = "Required column 'customer_id' not found"
        logger.error(error_msg)
        if strict:
            raise ValidationError(error_msg)
        return False

    try:
        # Run all validation checks
        check_duplicates(df)
        check_missing_values(df)
        check_distributions(df)
        if "label" in df.columns:
            check_label_distribution(df)
        check_data_leakage(df)

        logger.info("=" * 60)
        logger.info("✓ All validation checks passed")
        logger.info("=" * 60)
        return True

    except ValidationError as e:
        logger.error("=" * 60)
        logger.error("✗ Validation failed")
        logger.error("=" * 60)
        if strict:
            raise
        return False


def main(features_path: Optional[str] = None) -> None:
    """
    Main function to validate features.

    Parameters
    ----------
    features_path : str, optional
        Path to features Parquet file. Defaults to data/processed/features.parquet.
    """
    if features_path is None:
        features_path = PROCESSED_DATA_DIR / "features.parquet"

    try:
        success = validate_features(features_path, strict=True)
        if not success:
            sys.exit(1)
    except ValidationError:
        sys.exit(1)


if __name__ == "__main__":
    import sys

    features_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(features_path)

