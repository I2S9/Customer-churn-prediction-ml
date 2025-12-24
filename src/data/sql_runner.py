"""SQL execution runner for local database operations."""

import logging
from pathlib import Path
from typing import Optional

import duckdb

from src.utils.paths import PROCESSED_DATA_DIR, SQL_DIR, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_sql(
    sql_path: str | Path,
    db_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Optional[duckdb.DuckDBPyConnection]:
    """
    Execute SQL script from file.

    Parameters
    ----------
    sql_path : str | Path
        Path to SQL file to execute.
    db_path : str | Path, optional
        Path to DuckDB database file. Defaults to data/processed/churn.duckdb.
    output_path : str | Path, optional
        Path to save query results. If None, results are not saved.

    Returns
    -------
    Optional[duckdb.DuckDBPyConnection]
        Database connection (if output_path is None, connection is closed).
    """
    sql_path = Path(sql_path)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    if db_path is None:
        db_path = PROCESSED_DATA_DIR / "churn.duckdb"
    else:
        db_path = Path(db_path)

    ensure_dir(db_path.parent)

    logger.info(f"Connecting to database: {db_path}")
    con = duckdb.connect(str(db_path))

    logger.info(f"Executing SQL script: {sql_path}")
    sql_content = sql_path.read_text(encoding="utf-8")
    con.execute(sql_content)

    logger.info("SQL script executed successfully")

    if output_path:
        logger.info(f"Saving results to: {output_path}")
        # Keep connection open for further operations
        return con
    else:
        con.close()
        return None


def execute_query(
    query: str,
    db_path: Optional[str | Path] = None,
    return_df: bool = True,
):
    """
    Execute a SQL query and return results.

    Parameters
    ----------
    query : str
        SQL query to execute.
    db_path : str | Path, optional
        Path to DuckDB database file. Defaults to data/processed/churn.duckdb.
    return_df : bool, default True
        If True, return results as pandas DataFrame. Otherwise return raw results.

    Returns
    -------
    pandas.DataFrame or list
        Query results.
    """
    if db_path is None:
        db_path = PROCESSED_DATA_DIR / "churn.duckdb"
    else:
        db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info(f"Executing query on database: {db_path}")
    con = duckdb.connect(str(db_path))

    try:
        if return_df:
            result = con.execute(query).df()
            logger.info(f"Query returned {len(result)} rows")
            return result
        else:
            result = con.execute(query).fetchall()
            logger.info(f"Query returned {len(result)} rows")
            return result
    finally:
        con.close()


def main(sql_file: Optional[str] = None) -> None:
    """
    Main function to run SQL scripts.

    Parameters
    ----------
    sql_file : str, optional
        Specific SQL file to run. If None, runs 01_build_base_tables.sql.
    """
    logger.info("=" * 60)
    logger.info("SQL Runner: Building Base Tables")
    logger.info("=" * 60)

    if sql_file:
        sql_path = Path(sql_file)
    else:
        sql_path = SQL_DIR / "01_build_base_tables.sql"

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    run_sql(sql_path)

    logger.info("=" * 60)
    logger.info("SQL execution completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys

    sql_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(sql_file)

