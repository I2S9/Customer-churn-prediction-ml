"""Compute features using SQL and export to Parquet."""

import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.data.sql_runner import execute_query, run_sql
from src.utils.paths import PROCESSED_DATA_DIR, SQL_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_features(
    db_path: Optional[str | Path] = None,
    sql_file: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Compute features using SQL script and return as DataFrame.

    Parameters
    ----------
    db_path : str | Path, optional
        Path to DuckDB database file. Defaults to data/processed/churn.duckdb.
    sql_file : str | Path, optional
        Path to SQL file. Defaults to sql/02_compute_features.sql.

    Returns
    -------
    pandas.DataFrame
        Features DataFrame with customer_id, label, and all features.
    """
    if db_path is None:
        db_path = PROCESSED_DATA_DIR / "churn.duckdb"
    else:
        db_path = Path(db_path)

    if sql_file is None:
        sql_file = SQL_DIR / "02_compute_features.sql"
    else:
        sql_file = Path(sql_file)

    logger.info("=" * 60)
    logger.info("Feature Computation: SQL -> DataFrame")
    logger.info("=" * 60)

    # Execute SQL script to compute features
    logger.info(f"Executing feature computation SQL: {sql_file}")
    run_sql(sql_file, db_path=db_path)

    # Query the features table
    logger.info("Extracting features from database")
    df = execute_query("SELECT * FROM features", db_path=db_path, return_df=True)

    logger.info(f"Computed features for {len(df):,} customers")
    logger.info(f"Features shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")

    return df


def export_features_to_parquet(
    df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Export features DataFrame to Parquet file.

    Parameters
    ----------
    df : pandas.DataFrame
        Features DataFrame to export.
    output_path : str | Path, optional
        Output path. Defaults to data/processed/features.parquet.

    Returns
    -------
    Path
        Path to exported Parquet file.
    """
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "features.parquet"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)

    logger.info(f"Exporting features to: {output_path}")
    df.to_parquet(output_path, index=False, engine="pyarrow")

    file_size = output_path.stat().st_size / 1024**2
    logger.info(f"Features exported successfully ({file_size:.2f} MB)")
    logger.info(f"Features shape: {df.shape}")

    return output_path


def main(
    db_path: Optional[str | Path] = None,
    sql_file: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> None:
    """
    Main function to compute features and export to Parquet.

    Parameters
    ----------
    db_path : str | Path, optional
        Path to DuckDB database file.
    sql_file : str | Path, optional
        Path to SQL file for feature computation.
    output_path : str | Path, optional
        Output path for Parquet file.
    """
    # Compute features
    df = compute_features(db_path=db_path, sql_file=sql_file)

    # Export to Parquet
    export_features_to_parquet(df, output_path=output_path)

    logger.info("=" * 60)
    logger.info("Feature computation completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    sql_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    main(db_path=db_path, sql_file=sql_file, output_path=output_path)

