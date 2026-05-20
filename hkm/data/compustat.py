"""Fetch quarterly Compustat balance-sheet data for dealers and comparison groups."""

from __future__ import annotations

import pandas as pd
import psycopg2.extensions

from hkm.data.wrds_connect import run_query
from hkm.utils import get_logger, wrds_connection

logger = get_logger(__name__)


def fetch_compustat_quarterly(
    gvkeys: list[str],
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
    conn: psycopg2.extensions.connection | None = None,
) -> pd.DataFrame:
    """Fetch quarterly Compustat data for the given GVKEYs.

    Args:
        gvkeys: List of 6-digit Compustat GVKEYs.
        start_date: Earliest datadate to include (ISO format).
        end_date: Latest datadate to include (ISO format).
        conn: Open psycopg2 connection, or None to open one internally.

    Returns:
        DataFrame with columns: gvkey (str), datadate (Timestamp), atq (float),
        ceqq (float), book_debt (float = atq - ceqq), fyearq (int), fqtr (int).
        Only rows where atq > 0 and ceqq is not null are returned.
    """
    if not gvkeys:
        logger.warning("fetch_compustat_quarterly called with empty gvkeys list")
        return pd.DataFrame(
            columns=["gvkey", "datadate", "atq", "ceqq", "book_debt", "fyearq", "fqtr"]
        )

    placeholders = ", ".join(f"'{g}'" for g in gvkeys)
    sql = f"""
        SELECT gvkey, datadate, atq, ceqq, fyearq, fqtr
        FROM comp.fundq
        WHERE gvkey IN ({placeholders})
          AND datadate BETWEEN '{start_date}' AND '{end_date}'
          AND datafmt = 'STD'
          AND indfmt = 'INDL'
          AND popsrc = 'D'
          AND consol = 'C'
          AND atq IS NOT NULL
          AND atq > 0
          AND ceqq IS NOT NULL
        ORDER BY gvkey, datadate
    """

    def _execute(c: psycopg2.extensions.connection) -> pd.DataFrame:
        df = run_query(sql, c)
        if df.empty:
            return pd.DataFrame(
                columns=["gvkey", "datadate", "atq", "ceqq", "book_debt", "fyearq", "fqtr"]
            )
        df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
        df["datadate"] = pd.to_datetime(df["datadate"])
        df["atq"] = df["atq"].astype(float)
        df["ceqq"] = df["ceqq"].astype(float)
        df["book_debt"] = df["atq"] - df["ceqq"]
        df["fyearq"] = df["fyearq"].astype(int)
        df["fqtr"] = df["fqtr"].astype(int)
        logger.info(
            "Compustat: fetched %d rows for %d GVKEYs, date range %s–%s",
            len(df),
            len(gvkeys),
            start_date,
            end_date,
        )
        return df

    if conn is not None:
        return _execute(conn)
    with wrds_connection() as c:
        return _execute(c)


def fetch_compustat_all_quarterly(
    sic_filter: str | None = None,
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
    conn: psycopg2.extensions.connection | None = None,
) -> pd.DataFrame:
    """Fetch quarterly Compustat data for comparison groups (BD, Banks, all firms).

    Args:
        sic_filter: 'BD' for SIC 6211/6221, 'Banks' for SIC 6000–6299, None for all.
        start_date: Earliest datadate to include (ISO format).
        end_date: Latest datadate to include (ISO format).
        conn: Open psycopg2 connection, or None to open one internally.

    Returns:
        DataFrame with columns: gvkey (str), datadate (Timestamp), atq (float),
        ceqq (float), sich (str), book_debt (float).
    """
    # SIC codes live in comp.names (column: sic), not in comp.fundq.
    # Join fundq with names to filter by SIC.
    if sic_filter == "BD":
        sic_clause = "AND n.sic IN ('6211', '6221')"
    elif sic_filter == "Banks":
        sic_clause = (
            "AND n.sic IS NOT NULL AND n.sic ~ '^[0-9]+$' "
            "AND CAST(n.sic AS INTEGER) BETWEEN 6000 AND 6299"
        )
    else:
        sic_clause = ""

    sql = f"""
        SELECT q.gvkey, q.datadate, q.atq, q.ceqq, n.sic AS sich, q.fyearq, q.fqtr
        FROM comp.fundq q
        JOIN comp.names n ON q.gvkey = n.gvkey
        WHERE q.datadate BETWEEN '{start_date}' AND '{end_date}'
          AND q.datafmt = 'STD'
          AND q.indfmt = 'INDL'
          AND q.popsrc = 'D'
          AND q.consol = 'C'
          AND q.atq IS NOT NULL
          AND q.atq > 0
          AND q.ceqq IS NOT NULL
          {sic_clause}
        ORDER BY q.gvkey, q.datadate
    """

    def _execute(c: psycopg2.extensions.connection) -> pd.DataFrame:
        df = run_query(sql, c)
        if df.empty:
            return pd.DataFrame(
                columns=["gvkey", "datadate", "atq", "ceqq", "sich", "book_debt", "fyearq", "fqtr"]
            )
        df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
        df["datadate"] = pd.to_datetime(df["datadate"])
        df["atq"] = df["atq"].astype(float)
        df["ceqq"] = df["ceqq"].astype(float)
        df["sich"] = df["sich"].fillna("").astype(str)
        df["book_debt"] = df["atq"] - df["ceqq"]
        df = df.dropna(subset=["fyearq", "fqtr"])
        df["fyearq"] = df["fyearq"].astype(int)
        df["fqtr"] = df["fqtr"].astype(int)
        filter_label = sic_filter if sic_filter else "All"
        logger.info(
            "Compustat all: fetched %d rows, sic_filter=%s, date range %s–%s",
            len(df),
            filter_label,
            start_date,
            end_date,
        )
        return df

    if conn is not None:
        return _execute(conn)
    with wrds_connection() as c:
        return _execute(c)
