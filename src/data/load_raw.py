"""Load and validate raw data, save to staging format."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_raw_data_file(raw_dir: Path) -> Optional[Path]:
    """Find the first CSV or Parquet file in the raw data directory."""
    for ext in ["*.csv", "*.parquet"]:
        files = list(raw_dir.glob(ext))
        if files:
            return files[0]
    return None


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw data from CSV or Parquet file."""
    logger.info(f"Loading raw data from: {file_path}")

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate data structure: columns, types, and missing values."""
    logger.info("Validating data structure...")

    # Check basic shape
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Check columns
    logger.info(f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist())}")

    # Check data types
    logger.info("Data types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")

    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning("Missing values detected:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            logger.warning(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        logger.info("No missing values detected")

    # Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    else:
        logger.info("No duplicate rows detected")

    # Basic statistics
    logger.info("Basic statistics:")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def save_staging_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save data to staging Parquet format."""
    logger.info(f"Saving staging data to: {output_path}")

    ensure_dir(output_path.parent)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    file_size = output_path.stat().st_size / 1024**2
    logger.info(f"Staging data saved successfully ({file_size:.2f} MB)")


def main(raw_file: Optional[str] = None) -> None:
    """Main function to load raw data, validate, and save to staging."""
    logger.info("=" * 60)
    logger.info("Data Ingestion: Raw -> Staging")
    logger.info("=" * 60)

    # Find or use specified raw data file
    if raw_file:
        raw_path = Path(raw_file)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    else:
        raw_path = find_raw_data_file(RAW_DATA_DIR)
        if raw_path is None:
            raise FileNotFoundError(
                f"No CSV or Parquet file found in {RAW_DATA_DIR}"
            )

    # Load raw data
    df = load_raw_data(raw_path)

    # Validate data
    validate_data(df)

    # Save to staging
    staging_path = PROCESSED_DATA_DIR / "staging.parquet"
    save_staging_data(df, staging_path)

    logger.info("=" * 60)
    logger.info("Data ingestion completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys

    raw_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(raw_file)

