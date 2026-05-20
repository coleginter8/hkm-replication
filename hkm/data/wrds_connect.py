"""Thin psycopg2 wrapper: run SQL queries against WRDS and return DataFrames."""

from __future__ import annotations

import pandas as pd
import psycopg2.extensions

from hkm.utils import get_logger

logger = get_logger(__name__)


def run_query(sql: str, conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Execute a SQL query against the WRDS database and return results as DataFrame.

    Args:
        sql: The SQL string to execute.
        conn: An open psycopg2 connection.

    Returns:
        pd.DataFrame with column names from the cursor description.
    """
    preview = sql.strip()[:200].replace("\n", " ")
    logger.info("Running query: %s", preview)

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in (cur.description or [])]

    df = pd.DataFrame(rows, columns=columns)
    logger.info("Query returned %d rows", len(df))
    if len(df) == 0:
        logger.warning("Query returned zero rows: %s", preview)
    return df
