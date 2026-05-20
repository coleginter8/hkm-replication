"""Fetch monthly CRSP stock data, market index returns, and realized volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg2.extensions

from hkm.data.wrds_connect import run_query
from hkm.utils import get_logger, wrds_connection

logger = get_logger(__name__)


def fetch_crsp_monthly(
    permnos: list[int],
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
    conn: psycopg2.extensions.connection | None = None,
) -> pd.DataFrame:
    """Fetch monthly CRSP stock data for given PERMNOs.

    Args:
        permnos: List of CRSP PERMNOs.
        start_date: Earliest date (ISO format).
        end_date: Latest date (ISO format).
        conn: Open psycopg2 connection, or None to open one internally.

    Returns:
        DataFrame with columns: permno (int), date (Timestamp), prc (float),
        shrout (float), me (float = abs(prc) * shrout, in $ thousands).
    """
    if not permnos:
        logger.warning("fetch_crsp_monthly called with empty permnos list")
        return pd.DataFrame(columns=["permno", "date", "prc", "shrout", "me"])

    placeholders = ", ".join(str(p) for p in permnos)
    sql = f"""
        SELECT a.permno, a.date, a.prc, a.shrout
        FROM crsp.msf a
        WHERE a.permno IN ({placeholders})
          AND a.date BETWEEN '{start_date}' AND '{end_date}'
          AND a.prc IS NOT NULL
          AND a.shrout IS NOT NULL AND a.shrout > 0
        ORDER BY a.permno, a.date
    """

    def _execute(c: psycopg2.extensions.connection) -> pd.DataFrame:
        df = run_query(sql, c)
        if df.empty:
            return pd.DataFrame(columns=["permno", "date", "prc", "shrout", "me"])
        df["permno"] = df["permno"].astype(int)
        df["date"] = pd.to_datetime(df["date"])
        df["prc"] = df["prc"].astype(float)
        df["shrout"] = df["shrout"].astype(float)
        # ME in $thousands: |prc| * shrout (shrout is already in thousands of shares)
        df["me"] = np.abs(df["prc"]) * df["shrout"]
        logger.info(
            "CRSP monthly: fetched %d rows for %d PERMNOs, date range %s–%s",
            len(df),
            len(permnos),
            start_date,
            end_date,
        )
        return df

    if conn is not None:
        return _execute(conn)
    with wrds_connection() as c:
        return _execute(c)


def fetch_crsp_all_monthly(
    sic_codes: list[str] | None = None,
    sic_range: tuple[int, int] | None = None,
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
    conn: psycopg2.extensions.connection | None = None,
) -> pd.DataFrame:
    """Fetch monthly CRSP data for comparison groups filtered by SIC code.

    Args:
        sic_codes: Exact SIC codes (e.g. ['6211', '6221'] for broker-dealers).
        sic_range: (low, high) SIC range (e.g. (6000, 6299) for Banks).
        start_date: Earliest date (ISO format).
        end_date: Latest date (ISO format).
        conn: Open psycopg2 connection, or None to open one internally.

    Returns:
        DataFrame with columns: permno (int), date (Timestamp), prc (float),
        shrout (float), me (float), siccd (str).
        me is in $ thousands.
    """
    # crsp.msenames.siccd is an integer column in WRDS
    if sic_codes is not None:
        # Cast the string literals to integer for comparison
        sic_int_list = ", ".join(str(int(s)) for s in sic_codes)
        sic_clause = f"AND b.siccd IN ({sic_int_list})"
    elif sic_range is not None:
        lo, hi = sic_range
        sic_clause = f"AND b.siccd BETWEEN {lo} AND {hi}"
    else:
        sic_clause = ""  # all common stocks

    sql = f"""
        SELECT a.permno, a.date, a.prc, a.shrout, b.siccd
        FROM crsp.msf a
        JOIN crsp.msenames b ON a.permno = b.permno
          AND a.date BETWEEN b.namedt AND b.nameendt
        WHERE a.prc IS NOT NULL
          AND a.shrout IS NOT NULL AND a.shrout > 0
          AND a.date BETWEEN '{start_date}' AND '{end_date}'
          AND b.shrcd IN (10, 11)
          {sic_clause}
        ORDER BY a.permno, a.date
    """

    def _execute(c: psycopg2.extensions.connection) -> pd.DataFrame:
        df = run_query(sql, c)
        if df.empty:
            return pd.DataFrame(columns=["permno", "date", "prc", "shrout", "me", "siccd"])
        df["permno"] = df["permno"].astype(int)
        df["date"] = pd.to_datetime(df["date"])
        df["prc"] = df["prc"].astype(float)
        df["shrout"] = df["shrout"].astype(float)
        df["me"] = np.abs(df["prc"]) * df["shrout"]
        df["siccd"] = df["siccd"].astype(str)
        filter_label = (
            str(sic_codes) if sic_codes else str(sic_range) if sic_range else "All"
        )
        logger.info(
            "CRSP all monthly: fetched %d rows, sic=%s, date range %s–%s",
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


def fetch_crsp_market_index(
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
    conn: psycopg2.extensions.connection | None = None,
) -> pd.DataFrame:
    """Fetch CRSP monthly value-weighted market index returns.

    Args:
        start_date: Earliest date (ISO format).
        end_date: Latest date (ISO format).
        conn: Open psycopg2 connection, or None to open one internally.

    Returns:
        DataFrame with columns: date (Timestamp), vwretd (float).
        vwretd is the value-weighted return including dividends.
    """
    sql = f"""
        SELECT date, vwretd
        FROM crsp.msi
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
          AND vwretd IS NOT NULL
        ORDER BY date
    """

    def _execute(c: psycopg2.extensions.connection) -> pd.DataFrame:
        df = run_query(sql, c)
        df["date"] = pd.to_datetime(df["date"])
        df["vwretd"] = df["vwretd"].astype(float)
        logger.info("CRSP market index: fetched %d monthly rows", len(df))
        return df

    if conn is not None:
        return _execute(conn)
    with wrds_connection() as c:
        return _execute(c)


def fetch_crsp_daily_vol(
    start_date: str = "1970-01-01",
    end_date: str = "2012-12-31",
    conn: psycopg2.extensions.connection | None = None,
) -> pd.DataFrame:
    """Fetch realized quarterly market volatility from CRSP daily VW returns.

    Queries crsp.dsi for daily vwretd, then computes the quarterly standard
    deviation of daily log returns (annualized by sqrt(252)).

    Args:
        start_date: Earliest date (ISO format).
        end_date: Latest date (ISO format).
        conn: Open psycopg2 connection, or None to open one internally.

    Returns:
        DataFrame with columns: quarter (pd.Period[Q]), mkt_vol (float).
        mkt_vol is the annualized realized volatility (std dev * sqrt(252)).
    """
    sql = f"""
        SELECT date, vwretd
        FROM crsp.dsi
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
          AND vwretd IS NOT NULL
        ORDER BY date
    """

    def _execute(c: psycopg2.extensions.connection) -> pd.DataFrame:
        df = run_query(sql, c)
        df["date"] = pd.to_datetime(df["date"])
        df["vwretd"] = df["vwretd"].astype(float)

        # Daily log returns
        df["log_ret"] = np.log(1.0 + df["vwretd"])

        # Quarterly realized volatility (annualized)
        df["quarter"] = df["date"].dt.to_period("Q")
        vol: pd.DataFrame = (
            df.groupby("quarter")["log_ret"]
            .std()
            .rename("mkt_vol")
            .mul(np.sqrt(252))
            .reset_index()
        )
        logger.info(
            "CRSP daily vol: computed quarterly realized vol for %d quarters", len(vol)
        )
        return vol

    if conn is not None:
        return _execute(conn)
    with wrds_connection() as c:
        return _execute(c)
